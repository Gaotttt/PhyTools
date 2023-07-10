import pickle
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from .network import cfd_model, cloth_model, core_model
from .utils import cfd_eval, cloth_eval
from .dataset import dataset

PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}


class MeshGraphNets():

    def __init__(self, cfg):
        self.cfg = cfg

    def learner(self, model, params):
      """Run a learner job."""
      ds = dataset.load_dataset(self.cfg['dataset_dir'], 'train')
      ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
      ds = dataset.split_and_preprocess(ds, noise_field=params['field'],
                                        noise_scale=params['noise'],
                                        noise_gamma=params['gamma'])
      inputs = tf.data.make_one_shot_iterator(ds).get_next()

      loss_op = model.loss(inputs)
      global_step = tf.train.create_global_step()
      lr = tf.train.exponential_decay(learning_rate=1e-4,
                                      global_step=global_step,
                                      decay_steps=int(5e6),
                                      decay_rate=0.1) + 1e-6
      optimizer = tf.train.AdamOptimizer(learning_rate=lr)
      train_op = optimizer.minimize(loss_op, global_step=global_step)
      # Don't train for the first few steps, just accumulate normalization stats
      train_op = tf.cond(tf.less(global_step, 1000),
                         lambda: tf.group(tf.assign_add(global_step, 1)),
                         lambda: tf.group(train_op))

      with tf.train.MonitoredTrainingSession(
          hooks=[tf.train.StopAtStepHook(last_step=self.cfg['num_training_steps'])],
          checkpoint_dir=self.cfg['checkpoint_dir'],
          save_checkpoint_secs=600) as sess:

        while not sess.should_stop():
          _, step, loss = sess.run([train_op, global_step, loss_op])
          if step % 1000 == 0:
            logging.info('Step %d: Loss %g', step, loss)
        logging.info('Training complete.')


    def evaluator(self, model, params):
      """Run a model rollout trajectory."""
      ds = dataset.load_dataset(self.cfg['dataset_dir'], self.cfg['rollout_split'])
      ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
      inputs = tf.data.make_one_shot_iterator(ds).get_next()
      scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs)
      tf.train.create_global_step()

      with tf.train.MonitoredTrainingSession(
          checkpoint_dir=self.cfg['checkpoint_dir'],
          save_checkpoint_secs=None,
          save_checkpoint_steps=None) as sess:
        trajectories = []
        scalars = []
        for traj_idx in range(self.cfg['num_rollouts']):
          logging.info('Rollout trajectory %d', traj_idx)
          scalar_data, traj_data = sess.run([scalar_op, traj_ops])
          trajectories.append(traj_data)
          scalars.append(scalar_data)
        for key in scalars[0]:
          logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
        with open(self.cfg['rollout_path'], 'wb') as fp:
          pickle.dump(trajectories, fp)


    def train(self):
      tf.enable_resource_variables()
      tf.disable_eager_execution()
      params = PARAMETERS[self.cfg["model"]]
      learned_model = core_model.EncodeProcessDecode(
          output_size=params['size'],
          latent_size=128,
          num_layers=2,
          message_passing_steps=15)
      model = params['model'].Model(learned_model)
      if self.cfg["mode"] == 'train':
        self.learner(model, params)
      elif self.cfg["mode"] == 'eval':
        self.evaluator(model, params)
