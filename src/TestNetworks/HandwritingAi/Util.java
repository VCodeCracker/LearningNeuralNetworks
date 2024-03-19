package TestNetworks.HandwritingAi;

import org.jblas.DoubleMatrix;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Stream;

/**
 * Util class for serialization and other stuff. Not the best way to organize
 * methods but code conventions are discarded here
 *
 */
public class Util {
    private static final String FILE_DATA_TO_CHECK = "data/train-images.idx3-ubyte";
    private static final String FILE_DATA_TO_CHECK_LABEL = "data/train-images.idx3-ubyte";


    private Util() {

    }

    private static final String FILE_SERIALIZATION = "net.ser";

    /**
     * Serializes object
     *
     * @param obj object to serialize
     * @throws IOException if anything strange with io occurred
     */
    public static void serialize(Object obj) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(FILE_SERIALIZATION);
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream)) {
            objectOutputStream.writeObject(obj);
            objectOutputStream.flush();
        }
        System.out.println("Serialized");
    }

    /**
     * Deserializes to object
     *
     * @return deserialized object
     * @throws IOException            if anything strange with io occurred
     * @throws ClassNotFoundException if class wasn't found :(
     */
    public static Object deserialize() throws IOException, ClassNotFoundException {
        Object obj;
        FileInputStream fileInputStream = new FileInputStream(FILE_SERIALIZATION);
        try (ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream)) {
            obj = objectInputStream.readObject();
        }
        return obj;
    }

    /**
     * Prints matrix
     * @param matrix
     */
    public static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) throws ClassNotFoundException, IOException {
        int index = 0;
        SigmoidNetworkForHandWriting net = (SigmoidNetworkForHandWriting) deserialize();
        MnistMatrix[] mnistMatrix = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        printMnistMatrix(mnistMatrix[index]);
        System.out.println(net.feedForward(getDoubleMatrixAtIndex(index,mnistMatrix)));
    }

    public static String doubleMatrixToString(DoubleMatrix dm) {
        StringBuilder sb = new StringBuilder();
        for (double d : dm.toArray()) {
            sb.append(d >= 0.5 ? 1 : 0).append(' ');
        }
        return sb.toString();
    }

    public static DoubleMatrix getDoubleMatrixAtIndex(int index, MnistMatrix[] mnistMatrices) {
        MnistMatrix workingMnistMatrix = mnistMatrices[index];
        double[][] buffer = new double[28][28];
        for (int i = 0; i < workingMnistMatrix.getNumberOfRows(); i++) {
            for (int j = 0; j < workingMnistMatrix.getNumberOfColumns(); j++) {
                buffer[i][j] = (double) workingMnistMatrix.getValue(i, j) / 225;
            }
        }
        return new DoubleMatrix(buffer);
    }


}
