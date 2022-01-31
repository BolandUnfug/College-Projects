public class OrderedLinked<T extends Comparable> extends DoubleLinked{
    DNode next;
    DNode previous;
    DNode current;
    
    public OrderedLinked(){
        start = null;
        end = null;
        len = 0;
        }
    /**
     * Inserts a new item once it finds an item that is larger than itself, or hits the end of the array
     * @param data the data to be inserted
     */
    public void insert(T data){
        current = start;
            //System.out.println(len);
            for(int i = 0; i < len; i++){
                //System.out.println(i);
                if(current.getNext() != null && current.getData().compareTo(data) <= 0){
                    // if there is a next value, and the current node is not greater than the current value, move previos current and next up 1
                    previous = current;
                    current = current.getNext();
                    next = current.getNext();
                }
                else if(current.getData().compareTo(data) > 0){
                    //else, if the current node data is greater than data,  create a new node and insert it before current, rerouting as needed
                    //System.out.println("triggered");
                    DNode new_node = new DNode<T>(data);
                    new_node.setPrevious(previous);
                    new_node.setNext(current);
                    previous.setNext(new_node);             
                }
                else { //if none of the above, add a new node to the end of the list.
                    //System.out.println("triggered end");
                    DNode new_node = new DNode<T>(data);
                    new_node.setPrevious(current);
                    new_node.setNext(null);
                    current.setNext(new_node); 
                    end = new_node;
                }
            }
            len++;
    }

    /**
     * Removes the first instance of the data. iterates through the list untill the specified information is located.
     * @param data the character to be removed
     * @return if the letter was found and removed or not
     */
    public boolean remove(T data){
        current = start;
            //System.out.println(len);
            
            for(int i = 0; i < len; i++){
                //System.out.println(i);
                if(current.getNext() != null && current.getData().compareTo(data) != 0){ //if the next node is not null and not data
                    previous = current;
                    current = current.getNext();
                    next = current.getNext();
                }
                else if(current.getData().compareTo(data) == 0){// if the node equals data
                    
                    if( i+1 < len && i != 0){ // if it is 
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


    public boolean search(T data){
        current = start;
            //System.out.println(len);
            
            for(int i = 0; i < len; i++){
                //System.out.println(i);
                if(current.getNext() != null && current.getData().compareTo(data) != 0){
                    previous = current;
                    current = current.getNext();
                    next = current.getNext();
                }
                else if(current.getNext() != null || current.getData().compareTo(data) == 0){                    
                    return true;
                }
            }
        return false;
    }

}
