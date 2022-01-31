public class TestOrderedLinked {
    public static void main(String[] args) {
        
        DoubleLinked list1 = new DoubleLinked();
        OrderedLinked list2 = new OrderedLinked();
        OrderedLinked list3 = new OrderedLinked();
        OrderedLinked list4 = new OrderedLinked();
        OrderedLinked list5 = new OrderedLinked();
        testDouble(list1);
        testDouble(list2);
        testOrdered(list3);
        testOrdered2(list4);
        testOrdered3(list5);
    }

    public static void testDouble(DoubleLinked linkedlist){
        System.out.println("Populating the list. this should print abcd");
        linkedlist.append('a');
        linkedlist.append('b');
        linkedlist.append('d');
        linkedlist.append('e');
        linkedlist.print();
    }

    public static void testOrdered(OrderedLinked linkedlist){
        System.out.println("Populating the list. this should print abde.");
        linkedlist.append('a');
        linkedlist.append('b');
        linkedlist.append('d');
        linkedlist.append('e');
        linkedlist.print();
        System.out.println("Inserting c and z. This should print abcdez.");
        linkedlist.insert('c');
        linkedlist.insert('z');
        linkedlist.print();
        System.out.println("searching for and removing z. this should print true, followed by abcde.");
        System.out.println(linkedlist.search('z'));
        linkedlist.remove('z');
        linkedlist.print();
    }

        public static void testOrdered2(OrderedLinked linkedlist){
            System.out.println("Populating the list. this should print abde.");
            linkedlist.append(1);
            linkedlist.append(2);
            linkedlist.append(4);
            linkedlist.append(5);
            linkedlist.print();
            System.out.println("Inserting c and z. This should print abcdez.");
            linkedlist.insert(3);
            linkedlist.insert(20);
            linkedlist.print();
            System.out.println("searching for and removing z. this should print true, followed by abcde.");
            System.out.println(linkedlist.search(20));
            linkedlist.remove(20);
            linkedlist.print();
        
    }
    public static void testOrdered3(OrderedLinked linkedlist){
        System.out.println("Populating the list. this should print apple banana dragonfruit eggplant.");
        linkedlist.append("apple");
        linkedlist.append("banana");
        linkedlist.append("dragonfruit");
        linkedlist.append("eggplant");
        linkedlist.print();
        System.out.println("Inserting coconut and zebra. This should print apple banana coconut dragonfruit eggplant zebra.");
        linkedlist.insert("coconut");
        linkedlist.insert("zebra");
        linkedlist.print();
        System.out.println("searching for and removing zebra. this should print true, followed by apple banana coconut dragonfruit eggplant.");
        System.out.println(linkedlist.search("zebra"));
        linkedlist.remove("zebra");
        linkedlist.print();
    
}
}
