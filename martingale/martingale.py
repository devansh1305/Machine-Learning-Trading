""""""
"""Assess a betting strategy.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Devansh Panirwala (replace with your name)  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: dpanirwala3 (replace with your User ID)  		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 903262441 (replace with your GT ID)  		  	   		  		 			  		 			 	 	 		 		 	
"""




import numpy as np
import matplotlib.pyplot as plt
def author():
    """  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The GT username of the student  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: str  		  	   		  		 			  		 			 	 	 		 		 	
    """
    return "dpanirwala3"  # replace tb34 with your Georgia Tech username.


def gtid():
    """  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The GT ID of the student  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: int  		  	   		  		 			  		 			 	 	 		 		 	
    """
    return 903262441  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  		 			  		 			 	 	 		 		 	

    :param win_prob: The probability of winning  		  	   		  		 			  		 			 	 	 		 		 	
    :type win_prob: float  		  	   		  		 			  		 			 	 	 		 		 	
    :return: The result of the spin.  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: bool  		  	   		  		 			  		 			 	 	 		 		 	
    """
    result = False
    r = np.random.random()
    if r <= win_prob:
        result = True
    return result


def simulator(win_prob, limit, DEBUG):
    winnings = np.full((1001), 80)
    win_sum = 0

    count = 0
    while win_sum < 80:
        won = False
        bet = 1
        while not won:
            if count >= 1001:
                return winnings
            winnings[count] = win_sum
            count += 1
            won = get_spin_result(win_prob)
            if won:
                win_sum += bet
            else:
                win_sum -= bet
                bet *= 2
                if limit:
                    if win_sum == -256:
                        winnings[count:] = win_sum
                        return winnings
                    if win_sum - bet < -256:
                        bet = 256 + win_sum

    return winnings


def experiment1_fig1(win_prob, DEBUG):
    plt.axis([0, 300, -256, 100])
    plt.title("fig 1 - 10 trials")
    plt.xlabel("Number of Spins")
    plt.ylabel("Winnings")

    for i in range(10):
        episode = simulator(win_prob, False, DEBUG)
        plt.plot(episode)

    plt.savefig("./images/figure1.png")
    plt.clf()


def experiment1_fig2(winnings, DEBUG):
    mean_array = np.mean(winnings, axis=0)
    std = np.std(winnings, axis=0)
    mean_pos_array, mean_neg_array = mean_array + std, mean_array - std

    plt.axis([0, 300, -256, 100])
    plt.title("fig 2 - Mean of 1000 trials")
    plt.xlabel("Number of Spins")
    plt.ylabel("Winnings")

    plt.plot(mean_array, label="mean")
    plt.plot(mean_pos_array, label="mean+std")
    plt.plot(mean_neg_array, label="mean-std")

    plt.legend()
    plt.savefig("./images/figure2.png")
    plt.clf()


def experiment1_fig3(winnings, DEBUG):
    median_array = np.median(winnings, axis=0)
    std = np.std(winnings, axis=0)
    median_pos_array, median_neg_array = median_array + std, median_array - std

    plt.axis([0, 300, -256, 100])
    plt.title("fig 3 - Median of 1000 trials")
    plt.xlabel("Number of Spins")
    plt.ylabel("Winnings")

    plt.plot(median_array, label="median")
    plt.plot(median_pos_array, label="median+std")
    plt.plot(median_neg_array, label="median-std")
    plt.legend()
    plt.savefig("./images/figure3.png")
    plt.clf()


def experiment1_fig2_and_fig3(win_prob, DEBUG):
    winnings = np.zeros((1000, 1001))
    for i in range(1000):
        episode = simulator(win_prob, False, DEBUG)
        winnings[i] = episode

    experiment1_fig2(winnings, DEBUG)
    experiment1_fig3(winnings, DEBUG)


def experiment2_fig4(winnings, DEBUG):
    mean_array = np.mean(winnings, axis=0)
    std = np.std(winnings, axis=0)
    mean_pos_array, mean_neg_array = mean_array + std, mean_array - std

    plt.axis([0, 300, -256, 100])
    plt.title("fig 4 - Mean of 1000 trials with bankroll")
    plt.xlabel("Number of Spins")
    plt.ylabel("Winnings")

    plt.plot(mean_array, label="mean")
    plt.plot(mean_pos_array, label="mean+std")
    plt.plot(mean_neg_array, label="mean-std")

    plt.legend()
    plt.savefig("./images/figure4.png")
    plt.clf()


def experiment2_fig5(winnings, DEBUG):
    median_array = np.median(winnings, axis=0)
    std = np.std(winnings, axis=0)
    median_pos_array, median_neg_array = median_array + std, median_array - std

    plt.axis([0, 300, -256, 100])
    plt.title("fig 5 - Median of 1000 trials with bankroll")
    plt.xlabel("Number of Spins")
    plt.ylabel("Winnings")

    plt.plot(median_array, label="median")
    plt.plot(median_pos_array, label="median+std")
    plt.plot(median_neg_array, label="median-std")
    plt.legend()
    plt.savefig("./images/figure5.png")
    plt.clf()


def experiment2_fig4_and_fig5(win_prob, DEBUG):
    winnings = np.zeros((1000, 1001))
    count = 0
    for i in range(1000):
        episode = simulator(win_prob, True, DEBUG)
        winnings[i] = episode
    experiment2_fig4(winnings, DEBUG)
    experiment2_fig5(winnings, DEBUG)


def test_code():
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Method to test your code  		  	   		  		 			  		 			 	 	 		 		 	
    """
    DEBUG = False
    win_prob = 18/38  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    plt.style.use('bmh')
    experiment1_fig1(win_prob, DEBUG)
    experiment1_fig2_and_fig3(win_prob, DEBUG)
    experiment2_fig4_and_fig5(win_prob, DEBUG)
    # add your code here to implement the experiments


if __name__ == "__main__":
    test_code()
