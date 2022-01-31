//IMPORTANT: This is just skeleton code to be helpful, you will need to change the main program to call timeStack() on a multiple large numbers to gain insight into how your data structure's performance scales.

import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;

public class ReverseRunTime {
    public static void main(String[] args) throws IOException{
        reversetest();
    }
    public static void reversetest() throws IOException{
        ComplexLinked test = new ComplexLinked();
        ComplexLinked reversedtest = new ComplexLinked();
        int size = 0;
        long measurement = 0;
        BufferedWriter ReverseDataFile = new BufferedWriter( new FileWriter("ReverseRunTime.csv"));
        for(int i = 0; i < 1000; i++){
            for(int f = 0; f < 100; f++){
                test.append('x');
            }
            size += 100;
            measurement = 0;
            for (int j = 0; j < 10; j++){
                long start = System.nanoTime();
                reversedtest = test.reverse();
                long end = System.nanoTime();
                //System.out.println("measureing test " + i + " version " + j);
                measurement += end-start;
            }
            //System.out.println("averaging previous 10 tests");
            measurement /=10;
            ReverseDataFile.write("" + size + ", " + measurement);
            ReverseDataFile.newLine();
        }
	ReverseDataFile.close();
    }
}



