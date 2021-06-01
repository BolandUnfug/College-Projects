 /**
  *  SIR: Object-Oriented implementation. This is the "top level" class that ties all of the pieces of the SIR model together and drives simulation. <br>
  *  <br>
  *  Compilation: javac SIRModel.java <br>
  *  <br>
  *  Execution: java SIRModel [boolean modify]<br>
  *  <br>
  *  Usage example: java SIRModel <br>
  *  <br>
  *  Animates an SIR model with the default simulation parameters:
  *      infectionRate = 0.166;
  *      recoveryRate = 0.033;
  *      populationDensity = 0.001;
  *      maskDensity = .75;
  *      maxDays = 365;
  *      mapSize = 100;
  *      frameDuration = 100;
  *  This creats a map 100x100 that runs at 10 frames per second(.
  *  The simulation runs for a maximum of 365 days, and each day a cell has a base 16% chance to get infected by an infected neighbor and a 3% chance to recover.
  *  Out of this map, roughly 1 in 1000 start infectous, and about 50% wear masks.<br>
  *  <br>
  *  Usage example: java SIRModel [can literally be anything] <br>
  *  <br>
  *  Opens Console to set each of the custom values.
  *  <br>
  *  @author Boland Unfug, Caitrin Eaton, spring 2021
  */

  import java.util.Scanner;

public class SIRModel implements SIR {

    SIRMap map;
    SIRProbabilityMap probabilitymap;
    SIRScreen screen;
    SIRProbabilityScreen probabilityscreen;
    boolean probability = true;
    static SIRWriter file = new SIRWriter("");
    int generation;
    int maxGenerations;

    /**
     * Constructs a new SIR model.
     * @param mapSize (int) the width (and height) of the map, in cells
     * @param simLength (int) the maximum number of generations to simulate
     * @param populationDensity (double) the percent of cells that are initially occupied (SUSCEPTABLE)
     * @param frameDuration (long) the length of time (in milliseconds) for which each generation should be displayed on-screen
     */
    public SIRModel( double infectionRate, double recoveryRate,  double populationDensity,double maskdensity, int maxDays, int mapSize, int frameDuration ){
        // Initialize the map of cells
        if(probability == true){
            System.out.println("probability verification");
            probabilitymap = new SIRProbabilityMap( mapSize );
            probabilitymap.populate( populationDensity, maskdensity, 4 );
        }
        else{
            map = new SIRMap( mapSize );
            map.populate( populationDensity, maskdensity );
        }
        // Initialize the simulation time
        generation = 0;
        maxGenerations = maxDays;

        // Initialize the visualization screen in which the map of cells will appear
        if(probability == true){
            probabilityscreen = new SIRProbabilityScreen( probabilitymap );

            if(frameDuration == 0){
                probabilityscreen.setFrameTime();
            }
            else{
                probabilityscreen.setFrameTime( frameDuration );
            }
            
        }
        else{
            screen = new SIRScreen( map );
            if(frameDuration == 0){
                screen.setFrameTime();
            }
            else{
                screen.setFrameTime( frameDuration );
            }
        }
        
    }

    /**
     * Simulate a single generation.
     */
    public void update() {
        if(probability == true){
            generation++;
        int[] totals = {probabilitymap.getSusceptable(), probabilitymap.getInfected(), probabilitymap.getRecovered()};
        double total = probabilitymap.getSusceptable()+ probabilitymap.getInfected()+ probabilitymap.getRecovered();
        double[] percentages = {totals[0]/total, totals[1]/total,totals[2]/total};
        probabilitymap.update();
        file.update(generation, totals, percentages);
        probabilityscreen.update( generation);
        }
        else{
            generation++;
        int[] totals = {map.getSusceptable(), map.getInfected(), map.getRecovered()};
        double total = map.getSusceptable()+ map.getInfected()+ map.getRecovered();
        double[] percentages = {totals[0]/total, totals[1]/total,totals[2]/total};
        map.update();
        file.update(generation, totals, percentages);
        screen.update( generation);
        }
        
    }

    /**
     * Simulate the SIR model until either the cells stop changing (the map has zero infected cells) or the maximum
     * number of generations has been reached.
     */
    public void simulate() {
        generation = 0;
        if(probability == true){
            while( (probabilitymap.getInfected() > 0) && (generation < maxGenerations) ){
                update();
                
            }
        }
        while( (map.getInfected() > 0) && (generation < maxGenerations) ){
            update();
        }
    }

    /**
     * Runs a randomly initialized SIR model.
     * @param args String[], command line arguments (optional):
     *             args[0] = Any value, takes user to customization menu
     */
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        // Establish default simulation parameters, in case the user doesn't supply any command line arguments.
        double infectionRate = 0.166;
        double recoveryRate = 0.033;
        double populationDensity = 0.1;
        double maskDensity = .75;
        int maxDays = 365;
        int mapSize = 100;
        int frameDuration = 0;
        
        // allows user to customize simulation parameters if they input any command line arguments
        if (args.length > 0) {
            System.out.println("Input the infection rate (0-1)");
            infectionRate = input.nextDouble();
            System.out.println("Input the recovery rate (0-1)");
            recoveryRate = input.nextDouble();
            System.out.println("Input the population rate (0-1)");
            populationDensity = input.nextDouble();
            System.out.println("Input the mask density (0-1)");
            maskDensity = input.nextDouble();
            System.out.println("Input the max days (1-10000)");
            maxDays = input.nextInt();
            System.out.println("Input the map size (25-1000)");
            mapSize = input.nextInt();
            System.out.println("Input the frame duration (1-100)");
            frameDuration = input.nextInt();
        }
        
        
        file.open(maxDays, infectionRate, recoveryRate, mapSize);
        // Use manually input (or defualt) values to configure a new
        // SIR model, comprised of a set of SIRCells in a SIRMap and a SIRScreen display.
        SIR model = new SIRModel(infectionRate, recoveryRate, populationDensity, maskDensity, maxDays, mapSize,  frameDuration );
        
        // Run the simulation
        model.simulate();
        file.close();
    }
}