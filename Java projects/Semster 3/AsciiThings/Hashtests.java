public class Hashtests {
    public static void main(String[] args) {
        String word1 = "testing";
        String word2 = "this is also a test";
        char bad = (char) 0;
        word1 = word1 + bad;
        word2 = word2 + bad;
        System.out.println("word1 " + word1.hashCode());
        System.out.println("word1 " + word2.hashCode());
    }
}