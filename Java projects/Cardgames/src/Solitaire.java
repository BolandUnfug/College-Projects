
import java.util.*;
/**
The game of Solitaire
* Rules: https://bicyclecards.com/how-to-play/solitaire/#:~:text=The%20Tableau%3A%20Seven%20piles%20that,diamonds%2C%20spades%2C%20and%20clubs.
*note: There did not seem to be a rule for when the stock deck runs out, so I used the most common method of placing the waste deck back into the stock without shuffling
*Summary: the game of Solitaire. Move cards between piles in order to 'sort' the cards into each suite.
*
*@version 1.0
* @author Boland Unfug New College of Florida, 5/26/2021
*/
public class Solitaire extends CardGame {
    // Fields:
    private  int move = 0; // total number of moves
    private  Card selection; // the card currently being manipulated
    private  boolean victory; // if the player has won
    private Visual gamescreen; // game window
    private List<Card> discard; // the array list of the discard pile
    private List<Card> drawpile; // the array list of the draw pile
    private List<List<Card>> tableau; //the array list of 7 columns of cards in the tableau
    private List<List<Card>> foundation; // the array list of the 4 columns of cards in the foundation

    /**
     * constructor for Solitaire, creates a new solitaire game
     * @param none
     */
    public Solitaire(){
        this.move = 0;
        this.victory = false;
        this.gamescreen = new Visual();
        this.discard = new ArrayList<Card>();
        this.drawpile = new ArrayList<Card>();
        this.tableau = new ArrayList<List<Card>>(7);
        this.foundation = new ArrayList<List<Card>>(4);
    }
    /**
     * Runs the game of solitaire, functions as a callable main function
     * @param none
     */
    public void playSolitaire(){
        deck = shuffle(newDeck());
        deal(deck);
        while (victory != true){ // runs while player can continue playing
            //input action. can concede if the player thinks there are no more moves. could add an algorithm, but it would be complicated
            move++; // add move
            Visual.updateScreen(); // update the screen
            victoryCheck(); // check for victory
        }
    }
    /**
     * moves the top card of the draw pile to the discard pile. moves discard pile back to draw pile if draw pile is full.
     * @param none
     */
    public void drawCard(){
        //if deck empty, move discard back to drawpile and draw a card
        //here, selection is the top card of the draw pile
        moveCard(selection, drawpile, discard);
    }
    /**
     * moves the top card from discard to the appropriate foundation, if the conditions are met
     * <p>
     * conditions: there is a foundation pile with the same suite with the top card being 1 less than the selected card
     * @param none
     */
    public void discardtoFoundation(){
        //here, selection is the top card of the discard pile
        int foundationnum = 1; // this number would be the suite of the card
        moveCard(selection, discard, foundation.get(foundationnum));
    }
    /** 
     * moves the top card from the discard pile to the bottom of the selected column of the tableau, if the conditions are met
     * <p>
     * conditions: there is a tableau column with the bottom card being the opposite color and being 1 more than the selected card
     * @param tableauColumn int, one of seven columns within the tableau
    */
    public void discardtotableau(int tableauColumn){
        //here, selection is the top card of the discard pile
        moveCard(selection, discard, tableau.get(tableauColumn));
    }
    /** 
     * moves the card selected, as well as all cards below it, from one tableau column to another, as long as conditions are met
     * <p>
     * conditions:
     * <p>
     * there is a tableau pile with the opposite color with the bottom card being 1 more than the selected card
     * <p>
     * if there is an empty column, the only cards that can go there are kings
     * <p>
     * if there are no visible cards in a column, flip the bottommost card to be visible
     * <p>
     * if the bottom card is an ace, move it to the corresponding foundation
     * @param selection Card, the player specified card to move
     * @param tableauColumn1 int, the tableau column that card starts in
     * @param tableauColumn2 int, the tableau column the card is being moved to
    */
    public void tableautotableau(Card selection, int tableauColumn1, int tableauColumn2){
        //for number of cards in column, move 1 card starting from bottom of column until the selected card
        //here, selection is the selected card, and all cards below it
        moveCard(selection, tableau.get(tableauColumn1), tableau.get(tableauColumn2));
    }
    /** 
     * moves the bottom card of the selected tableau column to the correct foundation, if conditions are met
     * <p>
     * conditions:
     * <p>
     * there is a foundation pile with the same suite with the top card being 1 less than selected card
     * <p>
     * if the bottom card is an ace, move it to the corresponding foundation
     * @param tableauColumn int, the player specified tableau column
    */
    public void tableautoFoundation(int tableauColumn){
        //here, selection is the bottom card of the specified column in the tableau
        int foundationnum = 1;// this number would be the suite of the card
        moveCard(selection, tableau.get(tableauColumn), foundation.get(foundationnum));
    }
    /** 
     * deals the deck into the correct piles
     * <p>
     * @param deck Deck, the deck to be dealed
    */
    private void deal(Deck deck){
        
    }
    /** 
     * moves the card by removing it from the starting pile and adding it to the second
     * @see addCard
     * @see removeCard
     * @param card Card, the card to be moved
     * @param startpile List<Card>, the pile the card is starting in
     * @param endpile List<Card>, the pile the card is ending in
    */
    private void moveCard(Card card, List<Card> startpile, List<Card> endpile){

    }
    /** 
     * checks the victory conditions to see if the game is over
     * <p>
     * conditions:
     * <p>
     * if there are no further moves, the game ends. if the player has successfully 'sorted' the cards, they win.
     * @param none
    */
    private void victoryCheck(){
        victory = false;
    }
}
