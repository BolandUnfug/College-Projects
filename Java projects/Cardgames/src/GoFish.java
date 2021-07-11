
import java.util.*;
/**
 * Plays a game of go fish
 * Rules: https://bicyclecards.com/how-to-play/go-fish/
 * @author Boland Unfug, New College of Florida, 5/26/2021
 */
public class GoFish extends CardGame{
    //Fields:
    private List<List<Card>> hands; // a list of player hands
    private List<Integer> score; // a list of player scores
    private String input; // input
    private int playernum; // number of players
    private int turn; // the current turn
    private Card card; // temporary variable for card storage

    /**
     * the constructor for GoFish
     * @param playernum int, the number of players
     */
    public GoFish(int playernum){
        // if total player num is above 5, set to 5
        this.playernum = playernum;
        this.hands = new ArrayList<List<Card>>(playernum);
        this.score = new ArrayList<Integer>(playernum);
        this.turn = 0;
    }
    /**
     * acts as a main for the gofish class
     * <p>
     * runs a set of loops that keep the game going as long as there are cards in play. it then loops through each players turn,
     * and repeats the specified players turn until an opponent says go fish.
     * <p>
     * each turn, check hand for books, then ask an opponent for card. verify player 1 has card, as well as opponent. 
     * if opponent does nto have the card, draw a card and next turn. otherwise, swap the card between hands and repeat the turn.
     */
    public void playgofish(){
        deal(playernum, deck);
        // while there are cards in players hands or the deck
            for(int i = 0; i < playernum; i++){ // a loop that rotates through each players turn
                //while opponent has not said go fish
                    turn = i;
                    score.set(turn, score.get(turn) + bookcheck(turn));
                    //player 1 guesses a card that they have
                    //player 2 responds if they have the card or not, if they do move the card/cards
                    //computer verifies claim and response
                    verify(turn, card, 1);
                    //if it is verified, swap card/cards
                    cardswap(turn,card, 1);
                    // if player 2 does not have the card draw
                    carddraw(turn);
                    score.set(turn, score.get(turn) + bookcheck(turn)); //check hand for books, if there is a book remove it and add a point
            }
    }
    /** 
     * deals the 7 cards from the deck into each players hand, or 5 if there are 4 or 5 players
     * <p>
     * @param deck Deck, the deck to be dealed
     * @param playernum int, the number of players
    */
    private void deal(int playernum, Deck deck){
        //for playernum
        //create total number of scores and hands
        //deal 1 card to hands to determine starting player
        //shuffles cards back into deck and deals properly
    }
    /**
     * draws a card from the deck and gives it to the current player
     * @param turn int, the current turn, which decides the player
     */
    private void carddraw(int turn){

    }
    /**
     * swaps a card/cards between player's hands
     * @param turn int, the current turn, which decides the main player
     * @param card Card, the card being requested
     * @param opponent int, the opponent player 1 is asking for cards from
     */
    private void cardswap(int turn, Card card, int opponent){

    }
    /**
     * verifies that the players are telling the truth, ie asking for a card they have in hand and the opponent is not lying about posession of the card
     * @param turn int, the current turn, which decides the main player
     * @param card Card, the card being verified
     * @param opponent int, the opponent player 1 is asking for cards from
     */
    private void verify(int turn, Card card, int opponent){

    }
    /**
     * checks if the player has any books, and if so removes them and adds points.
     * @param turn int, the current turn, which decides the main player 
     * @return int, the number of books removed/ points
     */
    private int bookcheck(int turn){
        return 0; // temp value so there is no error
    }

}
