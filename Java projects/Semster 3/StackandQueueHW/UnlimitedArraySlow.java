/**
 * Works as a base for all of the core methods that do not change between UnlimitedArrayTypes.
 * @author Boland Unfug September 8th 2021
 * @version 0.3
 */
public class UnlimitedArraySlow extends UnlimitedArrayBase{
    /**
     * resize takes in an array of length n and increases the size. 1 version will double, one will add 5.
     * I actually learned something really interesting about arrays because of this.
     * Originally, I increased the size of the array by squaring the size of the previous array.
     * However, when I tested this with an array of 100,000, I got an out of bounds error, stating index 0 out of bouds for length 0.
     * turns out, the max data size for an array is 65536, and once the length hit one lower than that it would roll over to 0 and break.
     * TLDR; I made array size too large and learned after size 65536, it rolls over to 0 and breaks.
     * @param array the array to be resized
     * @return the new resized array
     */
    public int[] resize(int[] array){
        int[] newarray = new int[array.length+5];
        addslot = 0;
        removeslot = 0;

        for(int i = 0; i < array.length; i++){
            if(array[i] != 0 ){
                newarray[addslot] = array[i];
                addslot++;
            }
        }
        return newarray;
    }
}