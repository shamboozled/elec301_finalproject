import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class readJSONFile {
    public static double[] getJSONMatrixRow(String filepath, int rowIdx) throws IOException, ParseException {
        JSONParser parser = new JSONParser();

        JSONObject jsonMatrix = (JSONObject) parser.parse(new FileReader(filepath));
        int numCols = jsonMatrix.size();
        double[] row = new double[numCols];

        for (int i = 0; i < numCols; i++) {
            JSONObject jsonCol = (JSONObject) jsonMatrix.get(String.valueOf(i));
            try {
                row[i] = (double) jsonCol.get(String.valueOf(rowIdx));
            } catch (ClassCastException e) {
                row[i] = ((Long) jsonCol.get(String.valueOf(rowIdx))).doubleValue();
            }
        }

        return row;
    }

    public static double rowSum(double[] row) {
        double sum = 0;
        for (double elem : row) {
            sum += elem;
        }

        return sum;
    }

    public static void main(String args[]) throws IOException, ParseException {
        double[] result = getJSONMatrixRow("data.json", 0);
        System.out.println(rowSum(result));
    }
}
