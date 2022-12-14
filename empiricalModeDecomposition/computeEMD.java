import java.io.IOException;

public class computeEMD {
    public static void main(String[] args) throws IOException {
        double[][] matrix = readCSV.getCSVMatrix("data.csv", 1120, 253053);
        double[] data = matrix[3];

        EMD emd = new EMD();
        EMD.EmdData emdData = new EMD.EmdData();
        int order = 10;
        emd.emdCreate(emdData, data.length, order, 20, 0);
        emd.emdDecompose(emdData, data);

        // i is the data index, j is the IMF index
        for (int i=0;i<data.length;i++) {
            System.out.print(data[i]+";");
            for (int j=0;j<order; j++) System.out.print(emdData.imfs[j][i] + ";");
            System.out.println("\n ======== \n");
        }

    }
}
