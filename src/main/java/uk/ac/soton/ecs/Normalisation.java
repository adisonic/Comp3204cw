package uk.ac.soton.ecs;

/**
 * This class scales the features of a matrix represented as double[][] by columns.
 * 
 * Scaling to unit standard deviation and zero mean. The class remembers the standard
 * deviation and mean used to normalise and offers relevant getters and a convenience
 * method that normalises new vectors.
 */
public class Normalisation {
    private double[] stdev;
    private double[] mean;
    
    /**
     * This method changes the given array and returns the obtained means
     * and standard devaitions.
     *
     * The matrix is [row][colum] and it is normalised by each column.
     *
     * Will throw run-time errors if the matrix given is empty or the rows
     * are not of equal sizes.
     *
     * @param matrix Matrix to normalise.
     */
    public static Normalisation normalise(double[][] matrix) {
        int cols = matrix[0].length;
        int rows = matrix.length;

        System.out.println("COLUMNS = "+ cols);
        System.out.println("ROWS = "+ rows);

        double[] mean = new double[cols];
        double[] stdev = new double[cols];

        double sum;
        int i;
        int j;

        for (i=0;i<cols;i++) {
            sum = 0;
            for (j=0;j<rows;j++) {
                sum+=matrix[j][i];
            }
            mean[i] = sum/rows;
        }
        
        for (i=0;i<cols;i++) {
            sum=0;
            for (j=0;j<rows;j++) {
                matrix[j][i] -= mean[i];
                sum += matrix[j][i];
            }
            stdev[i] = Math.sqrt((sum*sum)/(rows-1));
        }

        for (i=0;i<cols;i++) {
            for (j=0;j<rows;j++) {
                matrix[j][i] /= stdev[i];
            }
        }

        Normalisation ret = new Normalisation(stdev, mean);
        return ret;
    }

    /**
     * The constructor is private because you never need to instantiate
     * this class unless you normalise a matrix by the normalise() method.
     */
    private Normalisation(double[] stdev, double[] mean) {
        this.stdev = stdev;
        this.mean = mean;
    }

    /**
     * Applies the same normalisation on the given item.
     */
    public void normaliseNewItem(double[] vector) {
        for (int j=0;j<vector.length;j++) {
            vector[j] = (vector[j]-mean[j])/stdev[j];
        }
    }

    /**
     * Get the standard deviations.
     */
    public double[] getStdev() {
        return this.stdev;
    }
    
    /**
     * Get the standard deviation for column index.
     * @param index The index.
     * @throws ArrayIndexOutOfBoundsException If the index is out of bounds.
     */
    public double getStdev(int index) {
        return this.stdev[index];
    }

    /**
     * Get the mean vector.
     */
    public double[] getMean() {
        return this.mean;
    }

    /**
     * Get the mean for colum index.
     * @param index The index.
     * @throws ArrayIndexOutOfBoundsException If the index is out of bounds.
     */
    public double getMean(int index) {
        return this.mean[index];
    }
}
