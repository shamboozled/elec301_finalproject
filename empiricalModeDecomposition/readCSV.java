import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class readCSV {
    public static double[][] getCSVMatrix (String filepath, int numSamples, int sampleSize) throws IOException {
        double[][] retval = new double[numSamples][sampleSize];

        int counter = 0;
        BufferedReader reader = new BufferedReader(new FileReader(filepath));
        for (String line = reader.readLine(); counter < numSamples; counter++) {
            line = reader.readLine();
            String[] stringArrayWithIndex = line.split(",");
            String[] stringArray = Arrays.copyOfRange(stringArrayWithIndex, 1, stringArrayWithIndex.length);
            double[] doubleRow = Arrays.stream(stringArray).mapToDouble(Double::parseDouble).toArray();
            retval[counter] = doubleRow;
        }

        return retval;
    }

    public static void main(String[] args) throws IOException {
        double[][] data = getCSVMatrix("data.csv", 5, 253053);
        System.out.println(Arrays.toString(data[1]));
    }
}
