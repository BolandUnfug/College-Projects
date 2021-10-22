/**
 * Modifies the resize method in order to resize the array more efficiently.
 * @author Boland Unfug September 8th 2021
 * @version 0.3
 */
public class UnlimitedArrayFast extends UnlimitedArrayBase{
    /**
     * resize takes in an array of length n and increases the size. 1 version will double, one will add 5.
     * @param array the array to be resized
     * @return the new resized array
     */
    public int[] resize(int[] array){
        int[] newarray = new int[array.length*3];
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