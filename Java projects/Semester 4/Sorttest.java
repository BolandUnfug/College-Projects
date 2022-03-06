import java.util.Random;
public class Sorttest{
    // generate a random set of numbers
    // take the largest and smallest value
    // get everything on a standardized scale, so that every value is unique
    // turn the position on the scale to a position in the array
    // place the value
        public static void main(String[] args) {
            int length = 100;
            Random rand = new Random();
            int[] test = new int[length];
            int smallest = length;

            for(int i = 0; i < length; i++){ //Generate a new instance
                int value = 99-i;
                if(value < smallest){
                    smallest = value;
                } 
                test[i] = value;  
            }

            System.out.println();
            int[] newlist = new int[length];
            
            for(int i = 0; i < length; i++){
                System.out.print(test[i] + ", ");
            }

            for(int i = 0; i < length; i++){
                
                newlist[test[i] - smallest] = test[i];
            }
        System.out.println("result I want: 1,2,3,4,5");
        for(int i = 0; i < length; i++){
            System.out.print(newlist[i] + ", ");
        }
    }
}