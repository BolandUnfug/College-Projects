import java.util.Random;

public class ParityMap {

    protected int size;
    protected int[] bitarray;
    protected int[][] bitmatrix;

    ParityMap(){
        this.size = 4;
        this.bitarray = new int[size*size];
        this.bitmatrix = new int[size][size];
        populate();
    }
    /**
     * Creates a new bitarray. randomly fills in non parity positions, then sets the parity positions based on the randomly generated bits.
     */
    public void populate(){
        for(int i = 0; i < bitarray.length; i++){
            Random rand = new Random();
            if(i != 0 && ((i & (i - 1)) != 0)){
               bitarray[i] = rand.nextInt(2); 
            }
            else{
                bitarray[i] = 0;
            }
        }
        //print();
        parityset(bitarray);
        //print();
        ArraytoMatrix();
    }

    public void corrupt(){
        Random rand = new Random();
        //bitarray[rand.nextInt(size*size)] ^= 1;
        bitarray[1] ^= 1;
        ArraytoMatrix();
    }

    public int[] parityset(int[] bitarray){
        int total = 0;
        int xor = 0;
        for(int i = 0; i < bitarray.length; i++){
            if(bitarray[i] == 1){
                total ++;
                bitarray[0] ^= 1;
                xor ^= i;
            }
        }
        int bit = xor;
        int bitlocation = 1;
        for(int j = 0; j < Math.sqrt(bitarray.length); j++){
            if(bit % 2 == 1){
                bitarray[0] ^= 1;
                bitarray[bitlocation] = 1;
            }
            bitlocation *= 2;
            bit /= 2;
        }
        //System.out.println();
        return bitarray;
    }

    public int finderror(){
        int xor = 0;
        for(int i = 0; i < bitarray.length; i++){
            if(bitarray[i] == 1){
                xor ^= i;
            }
        }
        bitarray[xor] += 2;
        ArraytoMatrix();
        return xor;
    }
    
    public void fix(int errorpos){
        bitarray[errorpos]-= 2;
        bitarray[errorpos] ^= 1;
        ArraytoMatrix();
    }

    private void ArraytoMatrix(){
        int row = 0;
        for(int i = 0; i < bitarray.length; i++){
            if(i%this.size == 0 && i != 0){
                row++;
            }
            bitmatrix[row][i-(this.size*row)] = bitarray[i];
            
        }
    }

    public void print(){
        for(int i = 0; i < bitarray.length; i++){
            System.out.print(bitarray[i]);
            if((i+1) % Math.sqrt(bitarray.length) == 0){
                System.out.println();
            }
        }
        System.out.println("-----");
    }

    public int[][] getMatrix(){
        int[][] extrabitmatrix = new int[this.size][this.size];
        for (int row=0; row<this.size; row++){
            for (int col=0; col<this.size; col++){
                extrabitmatrix[row][col] = this.bitmatrix[row][col];
            }
        }
        return extrabitmatrix;
    }

    public int getSize(){
        return this.size;
    }
}