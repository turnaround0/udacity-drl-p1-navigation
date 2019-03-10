from dqn_agent import DQNAgent
import torch.nn.functional as f


class DDQNAgent(DQNAgent):
    """
    Interacts with and learns from the environment.
    """

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # q_targets_next = max Q(next state, action, theta dash)
        # Not like DQN, action should be selected from local network instead of target network.
        # Select action to maximize Q values of Q local network in next states
        action_locals_next = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # Get Q values from Q target network by using selected action
        q_targets_next = self.qnetwork_target(next_states).gather(1, action_locals_next)

        # Compute Q targets for current states
        # If done, q_targets = rewards.
        # Otherwise, q_targets = rewards + gamma * q_targets_next
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_locals = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = f.mse_loss(q_locals, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Calculate gradients
        self.optimizer.step()       # Move to the gradients

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

