/**
 * not quite a radix sort, measure the longest length and create that many arrays, then sort all in one motion.
 */
public class Radix {
    private int[] arraytosort;
    private int highestnum;
    // allows the Radix array to take any or no input
    Radix (int[] array){
        this.arraytosort = array;
    }

    Radix(char[] array){

    }

    Radix(String[] array){

    }

    Radix(){
        System.out.println("error.");
    }


    private int[] sort(){
        int digits = findDigits();
        for(int i = 0; i < digits; i++){

        }
    }

    private int[] prefixSum(){

    }

    private int[] counts(){
        int[] countsarray = new int[10];
        for(int i = 0; i < arraytosort.length; i++){
            
        }
    }
    private int findDigits(){
        int digits = 0;
        for(int i = 0; i < arraytosort.length; i++){
            if(arraytosort[i]/10 > digits){
                digits = arraytosort[i]/10;
            }
        }
        return digits;
    }

}