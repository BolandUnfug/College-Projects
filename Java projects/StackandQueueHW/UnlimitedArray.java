/** This data structure stores a sequence of integers of arbitrary length
* It supports adding elements to the back-end of the sequence
* ... and removing elements from either the front or the back of the sequence
*
 * Ideas:
 * search array for first null to find the longest open space?
 * set the out variable equal to the in variiable when an array spot changes for last in first out.
 */

// IMPORTANT: Please include a constructor that takes no input
//          and creates an empty sequence with no data

public interface UnlimitedArray{
    /**
     * checks if there are any integers in the data structure
     * @return true if no integers, otherwise false
     */
    public boolean isEmpty();
    
    /**
     * getFirst returns the first number in the data structure. this is looking at the item that is least recently added.
     * @return first number in the sequence
     */
    public int getFirst();

    /**
    * getLast returns the integer which is the last number in the data structure. This is looking at the most-recently added number.
    * @return last number in the sequence
    */
    public int  getLast();

    /**
     * This adds a new number to the end [back-end] of the data structure.
     * @param number
     */
    public void add(int number);

    /**
     * This removes and returns the first number in the data structure. This will be the least-recently added number.
     * @return first number in the sequence.
     */
    public int removeFirst();

    /**
     * This removes and returns the last number in the data structure. This will be the most-recently added number.
     * @return Last number in the sequence.
     */
    public int removeLast();

}