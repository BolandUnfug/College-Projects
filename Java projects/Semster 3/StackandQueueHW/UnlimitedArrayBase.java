/**
 * Works as a base for all of the core methods that do not change between UnlimitedArrayTypes.
 * @author Boland Unfug September 8th 2021
 * @version 0.3
 */

class UnlimitedArrayBase implements UnlimitedArray{
    int[] numarray;
    int addslot = 0;
    int removeslot = 0;
    /**
     * Constructs an empty array of numbers
     */
    UnlimitedArrayBase(){
        numarray = new int[4];
    }
    
    public boolean isEmpty(){
        for(int i = 0; i < numarray.length; i++){
            if(numarray[i]  != 0){
                //System.out.println("is empty");
                return false;
            }
        }
        return true;
    }

    public int getFirst(){
        return removeslot;
    }

    public void add(int number){
        if (numarray.length - addslot != 0){
            numarray[addslot] = number;
            //System.out.println(numarray[addslot]);
            addslot += 1;
        }
        else{
            numarray = resize(numarray);
            add(number);
        }
    }
    public int getLast(){
        return addslot;
    }
    
    public int removeFirst(){
        int removednum = numarray[removeslot];
        numarray[removeslot] = 0;
        //System.out.println("why no do" + removednum);
        removeslot += 1;
        return removednum;
    }
    public int removeLast(){
        int removednum = numarray[addslot-1];
        numarray[addslot-1] = 0;
        //System.out.println("why no do" + removednum);
        addslot -=1;
        return removednum;
    }

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