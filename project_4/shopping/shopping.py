import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    def monthIndex(month):
        if month == "Jan": return 0
        elif month == "Feb": return 1
        elif month == "Mar": return 2
        elif month == "Apr": return 3
        elif month == "May": return 4
        elif month == "June": return 5
        elif month == "Jul": return 6
        elif month == "Aug": return 7
        elif month == "Sep": return 8
        elif month == "Oct": return 9
        elif month == "Nov": return 10
        else: return 11

    evicences = []
    labels = []

    with open(filename, mode="r") as shopping_csv:
        shopping_data = csv.DictReader(shopping_csv, delimiter=",")
        for row in shopping_data:
            row_evicences = []
            row_evicences.append( int(row["Administrative"]) )
            row_evicences.append( float(row["Administrative_Duration"]) )
            row_evicences.append( int(row["Informational"]) )
            row_evicences.append( float(row["Informational_Duration"]) )
            row_evicences.append( int(row["ProductRelated"]) )
            row_evicences.append( float(row["ProductRelated_Duration"]) )
            row_evicences.append( float(row["BounceRates"]) )
            row_evicences.append( float(row["ExitRates"]) )
            row_evicences.append( float(row["PageValues"]) )
            row_evicences.append( float(row["SpecialDay"]) )
            row_evicences.append( monthIndex(row["Month"]) )
            row_evicences.append( int(row["OperatingSystems"]) )
            row_evicences.append( int(row["Browser"]) )
            row_evicences.append( int(row["Region"]) )
            row_evicences.append( int(row["TrafficType"]) )
            row_evicences.append( 1 if row["VisitorType"] == "Returning_Visitor" else 0 )
            row_evicences.append( 1 if row["Weekend"] == "TRUE" else 0 )

            evicences.append( row_evicences )
            labels.append( 1 if row["Revenue"] == "TRUE" else 0 )

    return (evicences, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(1)
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    count = { 0: 0, 1: 0 }      # Counts the total positive/negative values
    true_count = { 0: 0, 1: 0 } # Counts the total true positive/negative predictions

    for expected, predicted in zip(labels, predictions):
        count[expected] += 1

        if expected == predicted: 
            true_count[expected] += 1

    return (true_count[1]/count[1], true_count[0]/count[0])


if __name__ == "__main__":
    main()
