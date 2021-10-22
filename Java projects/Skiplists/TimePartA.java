import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Random;

public class TimePartA{

    static long timeSkipList( int n, SkipList st){
		Random srand = new Random();
	
	
	for(int i=0; i < n; i++){
	    st.insert(i);
	}
	//This gets the start time in nanoseconds
	long start = System.nanoTime();
	for(int i = 0; i < n; i++){
		st.search(srand.nextInt(n));
	}
	//This gets the end time in nanoseconds
	long end = System.nanoTime();

	return end - start;
    }

    
    public static void main(String[] args) throws IOException{
	SkipList test_st;
	long measurement;
	int size;
	int iterations = 100;
	int samplesize = 1;
	int quantity = 100000;
	BufferedWriter slowDataFile = new BufferedWriter( new FileWriter("Skiplistdata.csv"));
	
	//------------------
    for(int i = 0; i < iterations; i++){
		//System.out.println(i);
		measurement = 0;
        size = i * quantity;
		for(int j = 0; j < samplesize; j++){
			test_st = new SkipList();
        	measurement += timeSkipList(size, test_st);
		}
        slowDataFile.write("" + size + ", " + measurement/samplesize);
        slowDataFile.newLine();
    }
	//-----------------

	slowDataFile.close();
    }
}