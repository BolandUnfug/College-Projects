import java.util.*;
/**
 * manages the general features of card games
 * @author Boland Unfug, New College of Florida, 5/26/2021
 */
public abstract class CardGame {
    //Fields
    protected Deck deck;
    /**
     * CardGame constructor, starts a new card game by creating a deck
     */
    CardGame(){
        this.deck = shuffle(newDeck());
    }
    /**
     * shuffles a deck
     * @param deck Deck, the unshuffled deck
     * @return Deck, the shuffled deck
     */
    protected Deck shuffle(Deck deck){
        return deck;
    }
    /**
     * draws a card from the deck
     * @param deck Deck, the deck to draw from
     * @return Card, the card drawn
     */
    protected Card draw(Deck deck){
        Card card = new Card(2, 'h'); // temp value for no errror
        return card;
    }
    /**
     * returns the top card of the deck
     * @return Card, the top card of the deck
     */
    protected Card topCard(){
        Card card = new Card(2, 'h'); // temp value for no errror
        return card;
    }
    /**
     * creates a new deck
     * @return Deck, a newly created deck
     */
    protected Deck newDeck(){
        Deck deck = new Deck();
        return deck;
    }
}
