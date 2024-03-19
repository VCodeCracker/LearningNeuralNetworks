package LearnNetworks;

import org.jblas.DoubleMatrix;

public class PerceptronNetworkSigmoidLearning {
    private int numLayers;
    private int[] sizes;
    private DoubleMatrix[] weights;
    private DoubleMatrix[] biases;



    public PerceptronNetworkSigmoidLearning(int... sizes) {
        this.sizes = sizes;
        numLayers = sizes.length;

        biases = new DoubleMatrix[sizes.length - 1];
        weights = new DoubleMatrix[sizes.length - 1];

        storeBiases();
        storeWeights();
    }
    public static void main(String[] args) {
        PerceptronNetworkSigmoidLearning net = new PerceptronNetworkSigmoidLearning(2,3,2);
        double[] inputs= {1,0};
        DoubleMatrix outputs = net.feedForward(new DoubleMatrix(inputs));

        System.out.println(outputs.toString());
    }
    public DoubleMatrix feedForward(DoubleMatrix a) {
    for (int i = 0; i < numLayers - 1; i++) {
        double[] z = new double[weights[i].rows];
        for (int j = 0; j < weights[i].rows; j++) {
            z[j] = weights[i].getRow(j).dot(a) + biases[i].get(j);
        }
        DoubleMatrix output = new DoubleMatrix(z);
        a = countSigmoid(output);
    }
    return a;
}
    private DoubleMatrix countSigmoid(DoubleMatrix z) {
        double[] output = new double[z.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = 1 / (1 + Math.exp(-z.get(i)));
        }
        return new DoubleMatrix(output);
    }
    private void storeBiases() {
        for (int i = 1; i < sizes.length; i++) {
            double[][] temp = new double[sizes[i]][];
            for (int j = 0; j < sizes[i]; j++) {
                double[] b = new double[] { 1 }; // Set to a constant value for a while
                temp[j] = b;
            }
            biases[i - 1] = new DoubleMatrix(temp);
        }
    }
    private void storeWeights() {
        for (int i = 1; i < sizes.length; i++) {
            double[][] temp = new double[sizes[i]][];
            for (int j = 0; j < sizes[i]; j++) {
                double[] w = new double[sizes[i - 1]];
                for (int k = 0; k < sizes[i - 1]; k++) {
                    w[k] = 0; // Set to a constant value for a while
                }
                temp[j] = w;
            }
            weights[i - 1] = new DoubleMatrix(temp);
        }
    }
}
