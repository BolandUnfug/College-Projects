// help from damien
// ways to improve: set connection branches into 1 set of options
// turn the connect previous and connect next into methods
import java.util.Random;

public class SkipList implements SearchList{
    private SNode<Comparable> current;
    private SNode<Comparable> start;
    private SNode<Comparable> end;
    private SNode<Comparable> below;
    private SNode<Comparable> above;
    private SNode<Comparable> sentinel;
    private int height;
   
    SkipList(){
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
        //System.out.println("------------------");
        //System.out.println("trying to insert " + data);
        Random rand = new Random();
        SNode newnode = new SNode<Comparable>(data);
        current = start;
        int tempheight = 1;






        
        if (end == null){ // if there are no nodes yet
            //System.out.println("needed a starting node. node being added: " + newnode);
            sentinel = new SNode<Comparable>(newnode);
            //System.out.println("sentinel position: " + sentinel);
            SNode currentsentinel = sentinel;
            end = newnode;
            current = end;
            start = sentinel;
            do{ // runs at least once, then runs until the coin flip is tails
                //System.out.print("#");
                // creating a new towernode for the first node
                SNode<Comparable> towernode = new SNode<Comparable>(data);
                towernode.setBelow(current);
                towernode.setAbove(null);
                current.setAbove(towernode);

                // creating a new sentinel tower node
                SNode<Comparable> sentineltowernode = new SNode<Comparable>(data);
                sentineltowernode.setBelow(currentsentinel);
                sentineltowernode.setAbove(null);
                currentsentinel.setAbove(sentineltowernode);

                // connecting tower and sentinel tower & updating start
                sentineltowernode.setNext(towernode);
                start = sentineltowernode;
                //updating current and currentsentinel
                current = towernode;
                currentsentinel = sentineltowernode;
                ////System.out.println("increased max height to " + height + 1);
                height ++;
            } while(rand.nextInt(2) != 0); // runs at least once, so that there is at least 1 tower node
            //System.out.println("");
        }




        else if (sentinel.getNext().getData().compareTo(data) >= 0){ // checks if data is smaller or equal to start
            // setting the new node to start, and changing start to the new node
            //System.out.println("data was before the start." + sentinel + " and after that " + sentinel.getNext());
            newnode.setNext(sentinel.getNext());
            newnode.setPrevious(null);
            SNode next = sentinel.getNext();
            sentinel.setNext(newnode);
            current = newnode;

            
            while(rand.nextInt(2) != 0){ //flips coin until it is 1
                // creating a new towernode, setting above and below
                //System.out.print("#");
                SNode<Comparable> towernode = new SNode<Comparable>(data);
                towernode.setBelow(current);
                towernode.setAbove(null);
                current.setAbove(towernode);
                towernode.setPrevious(null);
                current = current.getAbove();

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
            }
            
            //System.out.println("");
        }   
        else if (end.getData().compareTo(data) <= 0){ // checks if data is bigger or equal to end
             // setting the new node to start, and changing start to the new node
             //System.out.println("data was after the end");
             newnode.setPrevious(end);
             newnode.setNext(null);
             SNode previous = end;
             previous.setNext(newnode);
             end = newnode;
             current = newnode;
             
             while(rand.nextInt(2) != 0){ //flips coin until it is 1
                 // creating a new towernode, setting above and below
                 //System.out.print("#");
                 SNode<Comparable> towernode = new SNode<Comparable>(data);
                 towernode.setBelow(current);
                 towernode.setAbove(null);
                 current.setAbove(towernode);
                 towernode.setNext(null);
                 current = current.getAbove();
 
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
             }
             //System.out.println("");
        }
        else {
            // finding where the data should be inserted
            
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

            // setting up newnode, current and previous
            //System.out.println("data was in between.");
            SNode next = current.getNext();
            SNode previous = current;
            current = newnode;
            current.setPrevious(previous);
            current.setNext(next);
            next.setPrevious(current);
            previous.setNext(current);
            
            
            
            while(rand.nextInt(2) != 0){ //flips coin until it is 1
                // creating a new towernode, setting above and below
                //System.out.print("#");
                SNode<Comparable> towernode = new SNode<Comparable>(data);
                towernode.setBelow(current);
                towernode.setAbove(null);
                current.setAbove(towernode);
                current = current.getAbove();

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
        //printlist();
        //System.out.println("------------------");
        
    }





    //----------------------------------------------------------------------------------------------------------
    public boolean remove(Comparable data){ // THIS DOES NOT WORK. THIS IS ONLY HERE BECAUSE OF SEARCHLIST.
        current = start;
        boolean found = false;
        while(found == false){
            if(current.getNext() != null && current.getNext().getData().compareTo(data) <= 0){
                current = current.getNext();
            }
            else if(current.getBelow() != null){
                current = current.getBelow();
            }
            else if(current.getData().compareTo(data) == 0){
                
                if(current.getPrevious() == null){
                    SNode next = current.getNext();
                    next.setPrevious(null);
                    current.setNext(null);
                    while(current.getAbove() != null){
                    }
                }
                else if(current.getNext() == null){
                    SNode previous = current.getPrevious();

                }
                else{
                    SNode next = current.getNext();
                    SNode previous = current.getPrevious();

                }
                return true;
            }
            else{
                return false;
            }
        }
        return false;
    }

    //-------------------------------------------------------------------------------------------------------------------
    /**
     * searches the list for variable data
     * @param Comparable data, any data that can be numerically compared(int, char, string)
     * @return boolean, true if the item was found false otherwise
     */
    public boolean search(Comparable data){
        current = start;
        boolean found = false;
        while(found == false){
            if(current.getNext() != null && current.getNext().getData().compareTo(data) <= 0){
                current = current.getNext();
            }
            else if(current.getBelow() != null){
                current = current.getBelow();
            }
            else if(current.getData().compareTo(data) == 0){
                return true;
            }
            else{
                return false;
            }
        }
        return false;
    }

    public void printlist(){
        //Prints the contents of the list
        SNode<Comparable> current = sentinel.getNext();
        SNode<Comparable> base = sentinel.getNext();
        while (base.getData() != null){
            while( current.getAbove() != null){
                //System.out.print("#");
                current = current.getAbove();
            }
            //System.out.println(base.getData()+ " ");
            if(base.getNext() != null){
                base = base.getNext();
                current = base;
            }
            else{
                break;
            }
        }
        //System.out.println();
    }
}


