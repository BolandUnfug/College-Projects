public class BinaryTree {
    TNode start;

    BinaryTree(){
        start = null;
    }

    public TNode search(TNode head, Comparable data){
        if(head.getData().compareTo(data) > 0){
            if(head.getNextGreater() != null){
                head = head.getNextLesser();
                System.out.println("moved to the left. head is now " + head.getData());
                head = search(head, data);
            }
        }

        System.out.println("searching for " + data + " starting from " + head.getData());
        if(head.getNextLesser() != null){
            if(head.getData().compareTo(data) > 0){ // lesser
                
                
            }
        }

        else if (head.getNextGreater() != null){
            if(head.getData().compareTo(data) < 0){ // greater
                head = head.getNextGreater();
                System.out.println("moved to the right. head is now " + head.getData());
                head = search(head, data);
            }
        }
        System.out.println("returning " + head.getData());
        return head;
    }

    public void insert(Comparable data){
        System.out.println("----------------------------");
        if (start == null){
            start = new TNode(data);
            start.setAbove(null);
            start.setNextGreater(null);
            start.setNextLesser(null);
        }
        else{
            //System.out.println(data);
            TNode closest = search(start, data);
            //System.out.println("Closest Node: " + closest+ " data of " + closest.getData());
            TNode newbranch = new TNode (data);
            //System.out.println("new Node: " + newbranch+ " data of " + newbranch.getData());
            newbranch.setAbove(closest);
            if(closest.getData().compareTo(data) > 0){
                //System.out.println("node added to the left of " + closest.getData());
                closest.setNextLesser(newbranch);
            }
            else{
                //System.out.println("node added to the right of " + closest.getData());
                closest.setNextGreater(newbranch);
            }
            newbranch.setNextGreater(null);
            newbranch.setNextLesser(null);
        }
    }

    public void print(TNode n){
        //System.out.println(n);
        if(n.getNextGreater() != null){
            print(n.getNextGreater());
        }
        if (n.getNextLesser() != null){
            print(n.getNextLesser());
        }
    }

    public TNode getHighest(TNode n){
        if(n.getNextGreater() != null){
            n = getHighest(n.getNextGreater());
        }
        return n;
    }

    public int getHeight(TNode n, int counter){
        
        if((n.getNextGreater() != null  && getHeight(n.getNextGreater(), counter) > counter) 
        || (n.getNextLesser() != null  && getHeight(n.getNextLesser(), counter) > counter)){
            counter++;
        }
        System.out.println("counter:" + counter);
        return counter;
    }

    public Comparable getNextHigher(TNode n, Comparable data){
        TNode target = search(n, data);
        if(target.getNextGreater() != null){
            return target.getNextGreater().getData();
        }
        else if (target.getAbove() != null){
            return target.getAbove().getData();
        }
        return null;
    }

    
}

