package TestNetworks.CreateBinaryOFNumber;

import org.jblas.DoubleMatrix;

import java.lang.reflect.Array;
import java.util.Arrays;

public class SigmoidNetwork {
    private int numLayers;
    private int[] sizes;

    private DoubleMatrix[] weights;
    private DoubleMatrix[] biases;

    public SigmoidNetwork(int... sizes) {
        this.sizes = sizes;
        this.numLayers = sizes.length;

        this.biases = new DoubleMatrix[sizes.length - 1];
        this.weights = new DoubleMatrix[sizes.length - 1];

        storeWeights();
        storeBiases();

    }

    public static void main(String[] args) {
        SigmoidNetwork net = new SigmoidNetwork(10, 4);
        double[] inputs = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
        DoubleMatrix outputs = net.feedForward(new DoubleMatrix(inputs));

        System.out.println(outputs.toString("%.0f"));
    }

    private DoubleMatrix feedForward(DoubleMatrix a) {
        for (int i = 0; i < numLayers - 1; i++) {
            double[] z = new double[weights[i].rows];
            for (int j = 0; j < weights[i].rows; j++) {
                z[j] = weights[i].getRow(j).dot(a) + biases[i].get(j);
            }
            DoubleMatrix output = new DoubleMatrix(z);
            a = sigmoid(output);
        }
        return a;
    }
    private DoubleMatrix sigmoid(DoubleMatrix z) {
        double[] output = new double[z.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = 1 / (1 + Math.exp(-z.get(i)));
        }
        return new DoubleMatrix(output);
    }
    private void storeWeights() {
        final double  WEIGHT = 2;
        double[][] customWeights = new double[][] {
                {0, WEIGHT, 0, WEIGHT, 0, WEIGHT, 0, WEIGHT, 0, WEIGHT},
                {0, 0, WEIGHT, WEIGHT, 0, 0, WEIGHT, WEIGHT, 0, 0},
                {0, 0, 0, 0, WEIGHT, WEIGHT, WEIGHT, WEIGHT, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, WEIGHT, WEIGHT}
        };
        for (int i = 0; i < numLayers - 1; i++) {
            weights[i] = new DoubleMatrix(customWeights);
        }
    }
    private void storeBiases() {
        // Storing biases
        for (int i = 1; i < sizes.length; i++) {
            double[][] temp = new double[sizes[i]][];
            for (int j = 0; j < sizes[i]; j++) {
                double[] b = new double[] { -1 }; // Set to a constant value for a while
                temp[j] = b;
            }
            biases[i - 1] = new DoubleMatrix(temp);
        }
    }

}
