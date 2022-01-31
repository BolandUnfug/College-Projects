/**
 * This Data Structure implements a Queue of ints, First in First Out strucuture.
 * @author Boland Unfug, Scaffold provided by Matt Lapinski. September 3rd, 2021.
 */

public class Queue{
    UnlimitedArray data;

    /**
     * This constructor should be given an empty UnlimitedArray.
     * @param arr UnlimitedArray.
     */
    public Queue(UnlimitedArray arr){
        data = arr;
    }

    /**
     * Checks if the Queue is empty.
     * @return true if Queue is empty.
     */
    public boolean isEmpty(){
	    return data.isEmpty();
    }

    /**
     * Remove the least-recently added item from the Queue.
     * @return removed integer.
     */
    public int dequeue(){
	    int item = data.removeFirst();
	    return item;
    }

    /**
     * Add a new item to the Queue.
     * @param item to be added to Queue.
     */
    public void enqueue(int item){
	    data.add(item);
    }

    /**
     * Takes a peek at the next item that is ready to be dequeue'ed.
     * @return next int that is ready.
     */
    public int peek(){
	int item = data.getFirst();
	return item;
    }

    
}