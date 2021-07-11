/**
 * Defines an interface for SIR model file manipulation tools.
 * <br>
 * Saves information in the form of a csv file.
 * <br>
 *  @author Boland Unfug, Caitrin Eaton, spring 2021
 */
public interface Writer {

    /**
     * Create an output stream for recording an SIR model's daily demographics.
     * @param maxDays (int) maximum allowed duration of the simulation, in days.
     * @param infectionRate (double) the daily probability that an infectious individual will infect each susceptible neighbor
     * @param recoveryRate (double) the daily probability that an infectious individual will recover
     * @param size (int) number of individuals along each side of the "city" city
     */
    void open(int maxDays, double infectionRate, double recoveryRate, int size);

    /**
     * Record the day's results in the output file.
     * @param day (int) the current day in the simulation
     * @param totals (int[]) number of cells in each SIR State
     * @param percentages (double[]) percent of cells in each SIR State
     */
    void update(int day, int[] totals, double[] percentages);

    /**
     * Safely close the output file.
     */
    void close(); 
}
