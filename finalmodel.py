import gym
import pybullet_envs
import random
import datetime
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

tfd = tfp.distributions

tf.keras.backend.set_floatx('float64')

def build_actor(state_shape, action_shape, units=(400, 300, 100)):
    state = Input(shape=state_shape)
    x = Dense(units[0], name="L0", activation="relu")(state)
    for index in range(1, len(units)):
        x = Dense(units[index], name="L{}".format(index), activation="relu")(x)

    actions_mean = Dense(action_shape[0], name="Out_mean")(x)
    actions_std = Dense(action_shape[0], name="Out_std")(x)

    model = Model(inputs=state, outputs=[actions_mean, actions_std])

    return model


def build_critic(state_shape, action_shape, units=(400, 200, 100)):
    inputs = [Input(shape=state_shape), Input(shape=action_shape)]
    concat = Concatenate(axis=-1)(inputs)
    x = Dense(units[0], name="Hidden0", activation="relu")(concat)
    for index in range(1, len(units)):
        x = Dense(units[index], name="Hidden{}".format(index), activation="relu")(x)

    output = Dense(1, name="Out_QVal")(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)


class FinalModel:
    def __init__(
            self,
            env,
            lr_actor=3e-4,
            lr_critic=3e-4,
            actor_units=(256, 256),
            critic_units=(256, 256),
            auto_alpha=True,
            alpha=0.2,
            tau=0.005,
            gamma=0.98,
            batch_size=256,
            memory_cap=1000000,
            is_test=True
    ):
        self.env = env
        self.state_shape = env.observation_space.shape  # shape of observations
        self.action_shape = env.action_space.shape  # number of actions
        self.action_bound = (env.action_space.high - env.action_space.low) / 2
        self.action_shift = (env.action_space.high + env.action_space.low) / 2
        self.memory = deque(maxlen=int(memory_cap))

        self.actor = build_actor(self.state_shape, self.action_shape, actor_units)
        self.actor_optimizer = Adam(learning_rate=lr_actor)
        self.log_std_min = -20
        self.log_std_max = 2

        self.critic_1 = build_critic(self.state_shape, self.action_shape, critic_units)
        self.critic_target_1 = build_critic(self.state_shape, self.action_shape, critic_units)
        self.critic_optimizer_1 = Adam(learning_rate=lr_critic)
        update_target_weights(self.critic_1, self.critic_target_1, tau=1.)

        self.critic_2 = build_critic(self.state_shape, self.action_shape, critic_units)
        self.critic_target_2 = build_critic(self.state_shape, self.action_shape, critic_units)
        self.critic_optimizer_2 = Adam(learning_rate=lr_critic)
        update_target_weights(self.critic_2, self.critic_target_2, tau=1.)

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -np.prod(self.action_shape)
            self.log_alpha = tf.Variable(0., dtype=tf.float64)
            self.alpha = tf.Variable(0., dtype=tf.float64)
            self.alpha.assign(tf.exp(self.log_alpha))
            self.alpha_optimizer = Adam(learning_rate=lr_actor)
        else:
            self.alpha = tf.Variable(alpha, dtype=tf.float64)

        self.gamma = gamma  # discount factor
        self.tau = tau  # target model update
        self.batch_size = batch_size

        if is_test:
            self.load_actor("sac-saves/sac_actor_episode1100.h5")
            self.load_critic("sac-saves/sac_critic_episode1100.h5")

        # Tensorboard
        self.summaries = {}

    def process_actions(self, mean, log_std, test=False, eps=1e-6):
        std = tf.math.exp(log_std)
        raw_actions = mean

        if not test:
            raw_actions += tf.random.normal(shape=mean.shape, dtype=tf.float64) * std

        log_prob_u = tfd.Normal(loc=mean, scale=std).log_prob(raw_actions)
        actions = tf.math.tanh(raw_actions)

        log_prob = tf.reduce_sum(log_prob_u - tf.math.log(1 - actions ** 2 + eps))

        actions = actions * self.action_bound + self.action_shift

        return actions, log_prob

    def get_action(self, state, test=True, use_random=False):
        state = np.expand_dims(state, axis=0).astype(np.float64)

        if use_random:
            action = tf.random.uniform(shape=(1, self.action_shape[0]), minval=-1, maxval=1, dtype=tf.float64)
        else:
            means, log_stds = self.actor.predict(state)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)

            action, log_prob = self.process_actions(means, log_stds, test=test)

        q1 = self.critic_1.predict([state, action])[0][0]
        q2 = self.critic_2.predict([state, action])[0][0]
        self.summaries['q_min'] = tf.math.minimum(q1, q2)
        self.summaries['q_mean'] = np.mean([q1, q2])

        return action

    def save_model(self, actor_file_name, critic_file_name):
        self.actor.save(actor_file_name)
        self.critic_1.save(critic_file_name)

    def load_actor(self, actor_file_name):
        self.actor.load_weights(actor_file_name)
        print("actor loaded:{}".format(actor_file_name))
        print(self.actor.summary())

    def load_critic(self, critic_file_name):
        self.critic_1.load_weights(critic_file_name)
        self.critic_target_1.load_weights(critic_file_name)
        self.critic_2.load_weights(critic_file_name)
        self.critic_target_2.load_weights(critic_file_name)
        print("critic loaded:{}".format(critic_file_name))
        print(self.critic_1.summary())

    def remember(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        self.memory.append([state, action, reward, next_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        s = np.array(samples).T
        states, actions, rewards, next_states, dones = [np.vstack(s[i, :]).astype(np.float) for i in range(5)]

        with tf.GradientTape(persistent=True) as tape:
            means, log_stds = self.actor(next_states)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            next_actions, log_probs = self.process_actions(means, log_stds)

            current_q_1 = self.critic_1([states, actions])
            current_q_2 = self.critic_2([states, actions])
            next_q_1 = self.critic_target_1([next_states, next_actions])
            next_q_2 = self.critic_target_2([next_states, next_actions])
            next_q_min = tf.math.minimum(next_q_1, next_q_2)
            state_values = next_q_min - self.alpha * log_probs
            target_qs = tf.stop_gradient(rewards + state_values * self.gamma * (1. - dones))
            critic_loss_1 = tf.reduce_mean(0.5 * tf.math.square(current_q_1 - target_qs))
            critic_loss_2 = tf.reduce_mean(0.5 * tf.math.square(current_q_2 - target_qs))

            means, log_stds = self.actor(states)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            actions, log_probs = self.process_actions(means, log_stds)

            current_q_1 = self.critic_1([states, actions])
            current_q_2 = self.critic_2([states, actions])
            current_q_min = tf.math.minimum(current_q_1, current_q_2)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - current_q_min)

            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(
                    (self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)))

        critic_grad = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)  # compute actor gradient
        self.critic_optimizer_1.apply_gradients(zip(critic_grad, self.critic_1.trainable_variables))

        critic_grad = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)  # compute actor gradient
        self.critic_optimizer_2.apply_gradients(zip(critic_grad, self.critic_2.trainable_variables))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        self.summaries['q1_loss'] = critic_loss_1
        self.summaries['q2_loss'] = critic_loss_2
        self.summaries['actor_loss'] = actor_loss

        if self.auto_alpha:
            alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))
            self.summaries['alpha_loss'] = alpha_loss

    def train(self, max_epochs=8000, random_epochs=1000, max_steps=1000, save_freq=10):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        done, use_random, episode, steps, epoch, episode_reward = False, True, 0, 0, 0, 0
        cur_state = self.env.reset()

        while epoch < max_epochs:
            if steps > max_steps:
                done = True

            if done:
                episode += 1
                print("episode {}: {} total reward, {} alpha, {} steps, {} epochs".format(
                    episode, episode_reward, self.alpha.numpy(), steps, epoch))

                with summary_writer.as_default():
                    tf.summary.scalar('Main/episode_reward', episode_reward, step=episode)
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)

                summary_writer.flush()

                done, cur_state, steps, episode_reward = False, self.env.reset(), 0, 0
                if episode % save_freq == 0:
                    self.save_model("sac_actor_episode{}.h5".format(episode),
                                    "sac_critic_episode{}.h5".format(episode))

            if epoch > random_epochs and len(self.memory) > self.batch_size:
                use_random = False

            action = self.get_action(cur_state, use_random=use_random)
            next_state, reward, done, _ = self.env.step(np.clip(action[0], self.env.action_space.low, self.env.action_space.high))

            self.remember(cur_state, action, reward, next_state, done)
            self.replay()

            update_target_weights(self.critic_1, self.critic_target_1, tau=self.tau)
            update_target_weights(self.critic_2, self.critic_target_2, tau=self.tau)

            cur_state = next_state
            episode_reward += reward
            steps += 1
            epoch += 1

            with summary_writer.as_default():
                if len(self.memory) > self.batch_size:
                    tf.summary.scalar('Loss/actor_loss', self.summaries['actor_loss'], step=epoch)
                    tf.summary.scalar('Loss/q1_loss', self.summaries['q1_loss'], step=epoch)
                    tf.summary.scalar('Loss/q2_loss', self.summaries['q2_loss'], step=epoch)
                    if self.auto_alpha:
                        tf.summary.scalar('Loss/alpha_loss', self.summaries['alpha_loss'], step=epoch)

                tf.summary.scalar('Stats/alpha', self.alpha, step=epoch)
                if self.auto_alpha:
                    tf.summary.scalar('Stats/log_alpha', self.log_alpha, step=epoch)
                tf.summary.scalar('Stats/q_min', self.summaries['q_min'], step=epoch)
                tf.summary.scalar('Stats/q_mean', self.summaries['q_mean'], step=epoch)
                tf.summary.scalar('Main/step_reward', reward, step=epoch)

            summary_writer.flush()

        self.save_model("sac_actor_final_episode{}.h5".format(episode),
                        "sac_critic_final_episode{}.h5".format(episode))


if __name__ == "__main__":
    gym_env = gym.make("AntBulletEnv-v0")
    gym_env.render()
    model = FinalModel(gym_env)

    model.load_actor("sac_actor_episode1100.h5")
    model.load_critic("sac_critic_episode1100.h5")

    model.train(max_epochs=3000000, random_epochs=10000, save_freq=10)

