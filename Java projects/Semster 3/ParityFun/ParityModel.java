import java.util.Random;
import java.math.*;
public class ParityModel {

    public static void main(String[] args) {
        ParityMap bitMap = new ParityMap();
        bitMap.print();
        ParityVisualized screen = new ParityVisualized(bitMap);
        screen.update();
        
         bitMap.corrupt();
         int errorpos = bitMap.finderror();
         screen.update();
         bitMap.fix(errorpos);
         screen.update();
    }
    public static int[] sender(){ // simulates a computer creating data and sending it
        int[] bitarray = new int[16];
        return bitarray;

    }

    public void reciever(int[] bitarray){// simulates a computer recieving data and fixing it
        
    }

    
    
}