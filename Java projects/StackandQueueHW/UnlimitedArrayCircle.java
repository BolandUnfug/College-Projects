/**
 * not sure yet lol
 * @author Boland Unfug September 3rd 2021
 * @version 0.3
 */

class UnlimitedArrayCircle extends UnlimitedArrayBase{

    public int removeFirst(){
        if (numarray.length - removeslot != 0 && numarray[removeslot] != 0){
            int removednum = numarray[removeslot];
            numarray[removeslot] = 0;
            //System.out.println("why no do" + removednum);
            removeslot += 1;
            return removednum;
        }
        else {
            if (numarray[0] != 0){
                    removeslot = 0;
                    int removednum = numarray[removeslot];
                    numarray[removeslot] = 0;
                    removeslot += 1;
                    return removednum;
                }
                else{
                    return 0;
                }
                
        }
        
    }

    public void add(int number){
        if (numarray.length - addslot != 0 && numarray[addslot] == 0){
            numarray[addslot] = number;
            //System.out.println(numarray[addslot]);
            addslot += 1;
        }
        else{
            if(numarray[0] == 0){
                addslot = 0;
            }
            else{
                numarray = resize(numarray);
                add(number);
            }
        }
    }
    
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