import org.JMathStudio.DataStructure.Vector.Vector;
import org.JMathStudio.SignalToolkit.GeneralTools.SignalSpectrum;
import org.apache.commons.math3.complex.Complex;

import java.io.IOException;
import java.util.Arrays;

public class calculateEntropies {
    public static double[] entropiesForRow(double[] data) {
        // initialize the value to be returned
        double[] entropyVector = new double[40];

        // compute the top ten IMFs
        EMD emd = new EMD();
        EMD.EmdData emdData = new EMD.EmdData();
        int order = 10;
        emd.emdCreate(emdData, data.length, order, 20, 0);
        emd.emdDecompose(emdData, data);

        // modify the IMFs as covered in the paper
        double[][] modifiedIMFs = new double[8][data.length];
        // duplicate the first 6 IMFs
        System.arraycopy(emdData.imfs, 0, modifiedIMFs, 0, 6);
        // combine the residue and 4 remaining IMFs into 2 arrays
        modifiedIMFs[6] = addAndScaleVectors(emdData.imfs[6], emdData.imfs[7], true);
        modifiedIMFs[7] = addAndScaleVectors(addAndScaleVectors(emdData.imfs[8],
                emdData.imfs[9], true), emdData.residue, false);

        // calculate the different kinds of entropy for each of the eight IMF-sourced signals
        int counter = 0;
        for (double[] currentIMF : modifiedIMFs) {
            entropyVector[counter] = approximateEntropy(currentIMF);
            entropyVector[counter + 1] = sampleEntropy(currentIMF);
            entropyVector[counter + 2] = SVDEntropy(currentIMF);
            entropyVector[counter + 3] = spectralEntropy(currentIMF, 48000);
            entropyVector[counter + 4] = permutationEntropy(currentIMF);

            counter += 5;
        }

        return entropyVector;
    }

    public static double approximateEntropy(double[] u) {
        // initialize variables
        int m = 3;
        double r = 0.2 * std(u);
        int NSUM = u.length - m;

        double[] pos = new double[NSUM];
        double[] mat = new double[NSUM];
        double ratio = 0;
        double ApEn = 0;

        double posibles = 0;
        double matches = 0;
        double dif = 0, dif2 = 0, dif3 = 0;

        // ApEn calculation block
        for (int i = 1; i <= NSUM; i++) {
            posibles = 0;
            matches = 0;

            for (int j = 1; j <= NSUM; j++) {
                for (int k = 1; k <= m + 1; k++) {
                    // first possibility: it does not fit the condition for k < m
                    if (k < m) {
                        dif = Math.abs(u[i + k - 2] - u[j + k - 2]);
                        if (dif > r) {
                            break;
                        }
                    }

                    // second possibility: check if it is a possible block or not, k = m
                    if (k == m) {
                        dif2 = Math.abs(u[i + k - 2] - u[j + k - 2]);
                        if (dif2 > r) {
                            break;
                        } else {
                            posibles += 1;
                        }
                    }

                    // third possibility: check if it is a match block or not, k = m+1
                    if (k > m) {
                        dif3 = Math.abs(u[i + k - 2] - u[j + k - 2]);
                        if (dif3 > r) {
                            break;
                        } else {
                            matches += 1;
                        }
                    }
                }
            }

            pos[i - 1] = posibles;
            mat[i - 1] = matches;

        }

        // phi functions
        for (int i = 1; i <= NSUM; i++) {
            ratio = mat[i - 1] / pos[i - 1];
            ApEn += Math.log(ratio);
        }

        // final value of ApEn
        ApEn = (-1 / ((double) NSUM)) * ApEn;

        return ApEn;
    }

    public static double sampleEntropy(double[] u) {
        // initialize variables
        int m = 3;
        double r = 0.2 * std(u);
        int NSUM = u.length - m;

        double[] pos = new double[NSUM];
        double[] mat = new double[NSUM];
        double ratio = 0;
        double SampEn = 0;

        double posibles = 0;
        double matches = 0;
        double dif = 0, dif2 = 0, dif3 = 0;

        // ApEn calculation block
        for (int i = 1; i <= NSUM; i++) {
            posibles = 0;
            matches = 0;

            for (int j = 1; j <= NSUM; j++) {
                if (j != i) {
                    for (int k = 1; k <= m + 1; k++) {
                        // first possibility: it does not fit the condition for k < m
                        if (k < m) {
                            dif = Math.abs(u[i + k - 2] - u[j + k - 2]);
                            if (dif > r) {
                                break;
                            }
                        }

                        // second possibility: check if it is a possible block or not, k = m
                        if (k == m) {
                            dif2 = Math.abs(u[i + k - 2] - u[j + k - 2]);
                            if (dif2 > r) {
                                break;
                            } else {
                                posibles += 1;
                            }
                        }

                        // third possibility: check if it is a match block or not, k = m+1
                        if (k > m) {
                            dif3 = Math.abs(u[i + k - 2] - u[j + k - 2]);
                            if (dif3 > r) {
                                break;
                            } else {
                                matches += 1;
                            }
                        }
                    }
                }
            }

            pos[i - 1] = posibles;
            mat[i - 1] = matches;

        }

        // determine all total possibles and matches
        posibles = 0;
        matches = 0;

        for (int i = 1; i <= NSUM; i++) {
            posibles += pos[i -1];
            matches += mat[i - 1];
        }

        // SampEn calculation
        ratio = matches / posibles;
        SampEn = -Math.log(ratio);

        return SampEn;
    }

    public static double SVDEntropy(double[] data) {
        return 0;
    }

    public static double spectralEntropy(double[] data, int samplingFrequency) {
        // convert to float for jmathstudio methods
        float[] vector = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            vector[i] = (float) data[i];
        }
        Vector dataVector = new Vector(vector);

        // compute the PSD of the signal
        SignalSpectrum ss = new SignalSpectrum();
        Vector psd = ss.PSD(dataVector);

//        double[] psd = new double[samplingFrequency];
//        for (int i = 0; i < samplingFrequency; i++) {
//            psd[i] = Sxx(i, data, samplingFrequency);
//        }

        // calculate the normalization factor for the PSD
        double normFactor = 0;
        for (int i = 0; i < psd.length(); i++) {
            normFactor += psd.getElement(i);
        }

        // compute spectral entropy
        double SpecEn = 0;
        for (int i = 0; i < psd.length(); i++) {
            double sf = psd.getElement(i) / normFactor;
            SpecEn += sf * Math.log(sf) / Math.log(2);
        }

        if (Double.isNaN(SpecEn)) {
            SpecEn = 0;
        }

        return -1 * SpecEn;
    }

    public static double permutationEntropy(double[] data) {
        return 0;
    }

    private static double[] addAndScaleVectors(double[] op1, double[]op2, boolean scale) {
        double[] retval = new double[op1.length];

        if (scale) {
            for (int i = 0; i < op1.length; i++) {
                retval[i] = (op1[i] + op2[i]) * (1 / op1.length);
            }
        } else {
            for (int i = 0; i < op1.length; i++) {
                retval[i] = (op1[i] + op2[i]);
            }
        }


        return retval;
    }

    private static double std(double[] array) {

        // get the sum of array
        double sum = 0.0;
        for (double i : array) {
            sum += i;
        }

        // get the mean of array
        int length = array.length;
        double mean = sum / length;

        // calculate the standard deviation
        double standardDeviation = 0.0;
        for (double num : array) {
            standardDeviation += Math.pow(num - mean, 2);
        }

        return Math.sqrt(standardDeviation / length);
    }

    private static double Sxx(int freq, double[] signal, int Fs) {
        double t = (double) 1 / Fs;
        int T = signal.length;

        Complex s = new Complex(0, 0);
        for (int i = 0; i < T; i++) {
            Complex exponent = (Complex.I).multiply(-1 * 2 * Math.PI * freq * i * t);
            Complex factor = exponent.exp();
            s.add(factor.multiply(signal[i]));
        }

        return (Math.pow(s.abs(), 2) * Math.pow(t, 2)) / (T);
    }

    public static void main(String[] args) throws IOException {
//        double[] in1 = new double[]{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
//        double[] in2 = new double[]{0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1};
//        double[] in3 = new double[]{3, 4, 4, 3, 4, 3, 3, 4, 3, 3, 3, 4, 4, 3};
        double[][] data = readCSV.getCSVMatrix("data.csv", 5, 253053);
//        System.out.println(spectralEntropy(in1, 5));
//        System.out.println(spectralEntropy(in2, 5));
//        System.out.println(spectralEntropy(in3, 5));
        System.out.println(Arrays.toString(data[3]));
        System.out.println(approximateEntropy(data[3]));
    }
}
