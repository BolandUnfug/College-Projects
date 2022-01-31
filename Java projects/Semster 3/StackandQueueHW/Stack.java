/**
 * This Data Structure implements a Stack of ints, First in Last Out strucuture.
 * @author Boland Unfug, Scaffold provided by Matt Lapinski. September 3rd, 2021.
 */

public class Stack{
    UnlimitedArray data;

    /**
     * This constructor should be given an empty UnlimitedArray.
     * @param arr the empty array.
     */
    public Stack(UnlimitedArray arr){
        data = arr;
    }

    /**
     * Checks if the Stack is empty.
     * @return true if the Stack is empty, False otherwise.
     */
    public boolean isEmpty(){
	    return data.isEmpty();
    }
    
    /**
     * Removes the most-recently added item from the Stack.
     * @return the removed item.
     */
    public int pop(){
	    int item = data.removeLast();
	    return item;
    }

    /**
     * Add a new item to the Stack.
     * @param item the item to be added to the Stack.
     */
    public void push(int item){
	    data.add(item);
    }


    /**
     * Takes a peek at the next item that is ready to be pop'ed.
     * @return next int that is ready.
     */
    public int peek(){
	int item = data.getLast();
	return item;
    }

    
}