
/**
 * My extension of the Simple Linked List
 * Could this be used to create an array that can store different values, say an int and a string?
 * @author Boland Unfug, scaffold provided by Matt Lepinski
 * @version 2
 * @since 1.0 Fixed the reverse function
 */
public class ComplexLinked extends SimpleLinked{
    Node previous;
    Node current;
    Node next;

    public ComplexLinked(){
        start = null;
        len = 0;
    }

    /**
     * Inserts a new item at location, rerouts connections
     * @param location the location a new item is being inserted in
     * @param data the data to be inserted
     */
    public void insert(int location, char data){
        if (start == null){
            start = new Node(data);
            //System.out.println("needed a new node");
        }
        else{
            Node current = start;
            for(int i = 0; i < data; i++){
                if(i+1 == location){
                    current.setNext( new Node(data,current.getNext()) );
                }
                if(current.getNext() != null){
                    current = current.getNext();
                }
            }
        }
        //System.out.println("inserted " + data + " at " + location);
        len++;
    }
    
    /**
     * Removes the first instance of char data. iterates through the list untill the specified information is located.
     * @param data the character to be removed
     * @return if the letter was found and removed or not
     */
    public boolean remove(char data){
            current = start;
            //System.out.println(len);
            
            for(int i = 0; i < len; i++){
                //System.out.println(i);
                if(current.getNext() != null && current.getData() != data){
                    previous = current;
                    current = current.getNext();
                    next = current.getNext();
                }
                else if(current.getNext() != null || current.getData() == data){
                    
                    if( i+1 < len && i != 0){
                        previous.setNext(current.getNext());
                        current.setNext(null);
                    }
                    else if (i == 0){
                        //System.out.println("removed  " + current.getData());
                        start = current.getNext();
                    }
                    else{
                        //System.out.println("triggered");
                        previous.setNext(null);
                    }
                    current = start;
                    len--;
                    return true;
                }
            }
        return false;
    }

    /**
     * reverses a list and returns a new flipped version.
     * @return the new reversed ComplexLinked
     */
    public ComplexLinked reverse(){
        ComplexLinked reversed = new ComplexLinked();
        
        current = start;
        //System.out.println(start.getData());
        reversed.current = reversed.start;
        for(int i = 0; i < len; i++){
            //System.out.println(len);
            reversed.previous = reversed.current; //moves the previous node to the current node
            reversed.current = new Node(current.getData(),reversed.previous); // creates a new node for the reversed current node
            current = current.getNext(); // moves the original current node to the next node to copy
        }
        reversed.start = reversed.current;
        return reversed;
    }
}