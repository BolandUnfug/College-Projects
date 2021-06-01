/**
 * manages the playing of card games. this is for example purposes
 * @author Boland Unfug, New College of Florida, 5/26/2021
 */
public class Play {
    /**
     * runs the program
     * @param args
     */
    public static void main(String[] args){ // runs the code
        Solitaire game = new Solitaire();
        game.playSolitaire();
        GoFish game2 = new GoFish(3);
        game2.playgofish();
    }
}