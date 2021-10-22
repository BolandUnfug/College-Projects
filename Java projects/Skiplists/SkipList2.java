import java.util.Random; 
public class SkipList2 {
    private SNode<Comparable> current;
    private SNode<Comparable> start;
    private SNode<Comparable> end;
    private SNode<Comparable> below;
    private SNode<Comparable> above;
    private SNode<Comparable> sentinel;
    private int height;
   
    SkipList2(){
        sentinel = null;
        start = sentinel;
        end = null;
        height = 1;
    }


    /**
     * Inserts a new item once it finds an item that is larger than itself, or hits the end of the array
     * @param data the data to be inserted
     */
    public void insert(Comparable data){
        Random rand = new Random();
        SNode newnode = new SNode<Comparable>(data);
        SNode next = null;
        SNode  previous = null;
        boolean isnext = false;
        boolean isprevious = false;
        current = start;
        int tempheight = 1;


        if (end == null){ // if there are no nodes yet, create a sentinel node and connect them
            //System.out.println("needed a starting node. node being added: " + newnode);
            sentinel = new SNode<Comparable>(newnode);
            //System.out.println("sentinel position: " + sentinel);
            SNode currentsentinel = sentinel;
            end = newnode;
            current = end;
            start = sentinel;
        }

        else if(sentinel.getNext().getData().compareTo(data) >= 0){ // at the start
            isnext = true;
            newnode.setNext(sentinel.getNext());
            newnode.setPrevious(null);
            next = sentinel.getNext();
            sentinel.setNext(newnode);
            current = newnode;
        }

        else if (end.getData().compareTo(data) <= 0){ // at the end
            isprevious = true;
            newnode.setPrevious(end);
            newnode.setNext(null);
            previous = end;
            previous.setNext(newnode);
            end = newnode;
            current = newnode;
        }

        else{ // otherwise
            isnext = true;
            isprevious = true;
            boolean found = false;
            while(found == false){
                if(current.getNext() != null && current.getNext().getData().compareTo(data) <= 0){
                    current = current.getNext();
                }
                else if(current.getBelow() != null){
                    current = current.getBelow();
                }
                else if(current.getData().compareTo(data) <= 0 && current.getNext().getData().compareTo(data) >= 0){
                    found = true;
                }
            }
            
            next = current.getNext();
            previous = current;
            current = newnode;
            current.setPrevious(previous);
            current.setNext(next);
            next.setPrevious(current);
            previous.setNext(current);
        }

        while(rand.nextInt(2) != 0){ //flips coin until it is 1
            // creating a new towernode, setting above and below
            //System.out.print("#");
            SNode<Comparable> towernode = new SNode<Comparable>(data);
            towernode.setBelow(current);
            towernode.setAbove(null);
            current.setAbove(towernode);
            current = current.getAbove();

            if(isnext == true){
                // setting next connections
                if(next.getAbove() == null){ // if there is no tower above the next node
                    //move over until you find one
                    if (next.getNext() == null){ // if there is no node after the next node
                        current.setNext(null); // there is no towers of this height in the list, set next to null
                    }
                    else{ // if there is a node after the next node
                        while(next.getAbove() == null && next.getNext() != null){ // move the next node over until there is a node above
                            next = next.getNext();
                        }
                        current.setNext(next); // might get error here, if so add conditions
                    }
                }
                else{ // otherwise, connect to it and move next up.
                    current.setNext(next);
                    next = next.getAbove();
                }
            }
            if(isprevious == true){
                // setting previous connections
                if(previous.getAbove() == null){ // if there is no tower above the previoius node
                    //move over until you find one
                    if (previous.getPrevious() == null){ // if there is no node before the previous node
                        current.setPrevious(null); // there is no towers of this height in the list, set previous to null
                    }
                    else{ // if there is a node before the previous node
                        while(previous.getAbove() == null && previous.getPrevious() != null){ // move the previous node back until there is a node above
                            previous = previous.getPrevious();
                        }
                        current.setPrevious(previous); // might get a null error here, if so then add conditions
                    }
                }
                else{ // otherwise, connect to it and move previous up.
                    current.setPrevious(previous);
                    previous = previous.getAbove();
                }
            }
            
            
            if(tempheight > height){ // if a tower is made that is taller, add a sentinel
                SNode currentsentinel = start;
                // add a new sentineltowernode to the top of the sentinel
                SNode sentineltowernode = new SNode<Comparable>(current);
                sentineltowernode.setBelow(currentsentinel);
                sentineltowernode.setAbove(null);
                currentsentinel.setAbove(sentineltowernode);
                start = sentineltowernode;
                sentineltowernode.setNext(current);
            }
            //System.out.println("");
        }
    }
}
