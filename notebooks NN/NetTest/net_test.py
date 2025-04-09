import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

class Game:
    def __init__(self, payoff_matrix):
        self.payoff_matrix = payoff_matrix
        self.payoff_row_player = payoff_matrix[:, :, 0]
        self.payoff_col_player = payoff_matrix[:, :, 1]

    def get_payoff(self, row_action, col_action):
        return self.payoff_row_player[row_action, col_action], self.payoff_col_player[row_action, col_action]


class FictitiousPlayAgent:
    def __init__(self, game, num_actions, opponent=None):
        self.game = game
        self.num_actions = num_actions
        self.opponent = opponent
        
        # Initial belief: uniform strategy
        self.strategy = np.ones(num_actions) / num_actions
        self.strategy_history = [self.strategy.copy()]
        
        # Action history (for empirical distribution calculation)
        self.action_counts = np.zeros(num_actions)
        self.total_actions = 0
        
        self.payoff_history = []

    def play(self):
        if self.total_actions == 0:
            # First move: pick randomly
            action = np.random.choice(self.num_actions)
        else:
            # Compute empirical frequency of opponent's actions
            opponent_action_distribution = self.opponent.action_counts / self.opponent.total_actions

            # Compute best response (greedy selection)
            expected_payoffs = self.game.payoff_row_player @ opponent_action_distribution
            action = np.argmax(expected_payoffs)

        # Update action count
        self.action_counts[action] += 1
        self.total_actions += 1

        # Save strategy (empirical distribution of own actions)
        self.strategy = self.action_counts / self.total_actions
        self.strategy_history.append(self.strategy.copy())

        return action

    def update_payoff(self, payoff):
        self.payoff_history.append(payoff)

class BestResponseAgent:
    def __init__(self, game, num_actions, opponent=None):
        self.game = game
        self.num_actions = num_actions
        self.opponent = opponent
        self.action_counts = np.zeros(num_actions)
        self.total_actions = 0

        # Track opponent's action history
        self.opponent_action_counts = np.zeros(num_actions)
        self.total_observed_actions = 0

        self.payoff_history = []

        self.strategy = np.ones(num_actions) / num_actions
        self.strategy_history = [self.strategy.copy()]

    def play(self):
        if self.total_observed_actions == 0:
            # pick randomly
            action = np.random.choice(self.num_actions)
        else:
            # Identify the opponent’s most frequently played action
            most_frequent_action = np.argmax(self.opponent_action_counts)

            # Compute the best response to that action
            best_response = np.argmax(self.game.payoff_row_player[:, most_frequent_action])
            action = best_response

        # Update action count
        self.action_counts[action] += 1
        self.total_actions += 1

        # Save strategy (empirical distribution of own actions)
        self.strategy = self.action_counts / self.total_actions
        self.strategy_history.append(self.strategy.copy())

        return action

    def observe_opponent_action(self, action):
        # Update opponent action counts
        self.opponent_action_counts[action] += 1
        self.total_observed_actions += 1

    def update_payoff(self, payoff):
        self.payoff_history.append(payoff)

# neural network for predicting the divergence or convergence of the two agents
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x

def main():

    # prisoners dilemma
    # payoff_matrix = np.array([
    #     [[1, 1], [0, 3]],  # Player 1 cooperates, Player 2 cooperates
    #     [[3, 0], [2, 2]]   # Player 1 defects, Player 2 cooperates
    # ])


    model = NeuralNetwork(input_size=18, hidden_size=32, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    # Training the neural network
    num_epochs = 1000

    loss_history = []
    
    for epoch in range(num_epochs):

        #create random payoff matrix
        random_payoff_matrix = np.random.randint(-1, 1, (3, 3, 2)) 

        # Create the game
        g = Game(random_payoff_matrix)

        # Create the agents
        agent1 = BestResponseAgent(g, 3, opponent=None)
        agent2 = BestResponseAgent(g, 3, opponent=agent1)  # Connect agent2 to agent1
        agent1.opponent = agent2  # Connect agent1 to agent2

        # Play the game
        num_iterations = 2000
        previous_strategy = copy.deepcopy(agent1.strategy)
        previous_strategy2 = copy.deepcopy(agent2.strategy)
        for _ in range(num_iterations):

            previous_strategy = copy.deepcopy(agent1.strategy)
            previous_strategy2 = copy.deepcopy(agent2.strategy)

            action1 = agent1.play()
            action2 = agent2.play()

            payoff1, payoff2 = g.get_payoff(action1, action2)

            agent1.update_payoff(payoff1)
            agent2.update_payoff(payoff2)

            
        #check if strategies are converged or very close
        converged = False
        if np.array_equal(previous_strategy, agent1.strategy) and np.array_equal(previous_strategy2, agent2.strategy):
            converged = True
        elif np.all(np.abs(previous_strategy - agent1.strategy) < 1e-6) and np.all(np.abs(previous_strategy2 - agent2.strategy) < 1e-6):
            converged = True
        else:
            converged = False


        #flatten the payoff matrix for the neural network
        flat_payoff_matrix = torch.tensor(random_payoff_matrix.flatten(), dtype=torch.float32)

        # Create the target tensor (1 for convergence, 0 for divergence)
        target = torch.tensor([1.0 if converged else 0.0], dtype=torch.float32)

        # Forward pass
        optimizer.zero_grad()
        output = model(flat_payoff_matrix)
        loss = criterion(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print the loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

            loss_history.append(loss.item())

    # Plot the loss history
    

    #running avrage of the loss
    running_avg_loss = np.convolve(loss_history, np.ones(100)/100, mode='valid')
    plt.plot(running_avg_loss, label='Running Avg Loss', color='red')
    plt.plot(loss_history, label='Loss History', color='blue')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.savefig("loss_history.png")

    #test the network on various games

    correct_predictions = 0
    total_predictions = 0

    for i in range(100):
        #create random payoff matrix
        random_payoff_matrix = np.random.randint(-1, 1, (3, 3, 2)) 

        #flatten the payoff matrix for the neural network
        flat_payoff_matrix = torch.tensor(random_payoff_matrix.flatten(), dtype=torch.float32)

        # Forward pass
        output = model(flat_payoff_matrix)
        print(f"Test Output: {output.item()}")

        #play the game with agents to check convergence
        g = Game(random_payoff_matrix)
        agent1 = FictitiousPlayAgent(g, 3, opponent=None)
        agent2 = FictitiousPlayAgent(g, 3, opponent=agent1)  # Connect agent2 to agent1
        agent1.opponent = agent2  # Connect agent1 to agent2

        num_iterations = 2000
        previous_strategy = copy.deepcopy(agent1.strategy)
        previous_strategy2 = copy.deepcopy(agent2.strategy)
        for _ in range(num_iterations):
            
            previous_strategy = copy.deepcopy(agent1.strategy)
            previous_strategy2 = copy.deepcopy(agent2.strategy)

            action1 = agent1.play()
            action2 = agent2.play()

            payoff1, payoff2 = g.get_payoff(action1, action2)

            agent1.update_payoff(payoff1)
            agent2.update_payoff(payoff2)

        #check if strategies are converged or very close
        converged = False
        if np.array_equal(previous_strategy, agent1.strategy) and np.array_equal(previous_strategy2, agent2.strategy):
            converged = True
        elif np.all(np.abs(previous_strategy - agent1.strategy) < 1e-6) and np.all(np.abs(previous_strategy2 - agent2.strategy) < 1e-6):
            converged = True
        else:
            converged = False

        print(f"Converged: {converged}")

        # Compare the neural network output with the actual convergence
        if (output.item() >= 0.5 and converged) or (output.item() < 0.5 and not converged):
            correct_predictions += 1
        total_predictions += 1
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")

    #test the neural network on rock paper scissors
    test_payoff_matrix = np.array([
        [[0, 0], [1, -1], [-1, 1]],  
        [[-1, 1], [0, 0], [1, -1]], 
        [[1, -1], [-1, 1], [0, 0]] 
    ])

    test_flat_payoff_matrix = torch.tensor(test_payoff_matrix.flatten(), dtype=torch.float32)
    test_output = model(test_flat_payoff_matrix)
    print(f"Test Output: {test_output.item()}")

    #actual test with agents
    g = Game(test_payoff_matrix)
    agent1 = FictitiousPlayAgent(g, 3, opponent=None)
    agent2 = FictitiousPlayAgent(g, 3, opponent=agent1)  # Connect agent2 to agent1
    agent1.opponent = agent2  # Connect agent1 to agent2
    num_iterations = 1000
    for _ in range(num_iterations):
        action1 = agent1.play()
        action2 = agent2.play()

        payoff1, payoff2 = g.get_payoff(action1, action2)

        agent1.update_payoff(payoff1)
        agent2.update_payoff(payoff2)
    # Check if the two agents are converged to a Nash equilibrium
    if np.array_equal(agent1.strategy, agent2.strategy):
        converged = True
    else:
        converged = False
    print(f"Converged: {converged}")

    # Plot the strategy history for all actions
    plt.figure()
    plt.plot(agent1.strategy_history, label='Agent 1 Strategy')
    plt.plot(agent2.strategy_history, label='Agent 2 Strategy')
    plt.xlabel("Iterations")
    plt.ylabel("Strategy")
    plt.title("Strategy History")
    plt.legend()
    plt.savefig("strategy_history.png")
    plt.show()

if __name__ == "__main__":
    main()
