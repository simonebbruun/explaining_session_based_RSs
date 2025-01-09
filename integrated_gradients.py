import tensorflow as tf


def interpolate_instances(baseline, instance, alphas):
  alphas_x = alphas[:, tf.newaxis, tf.newaxis]
  delta = instance - baseline
  instances = baseline +  alphas_x * delta
  return instances


def compute_gradients(instances, model, target_class_idx):
  with tf.GradientTape() as tape:
    tape.watch(instances)
    logits = model(instances)
    # probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    probs = logits[:, target_class_idx]
  return tape.gradient(probs, instances)


def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients


def integrated_gradients(baseline, instance, model, target_class_idx, m_steps=50, batch_size=32):
  # Generate alphas.
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

  # Collect gradients.    
  gradient_batches = []

  # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

    gradient_batch = one_batch(baseline, instance, alpha_batch, model, target_class_idx)
    gradient_batches.append(gradient_batch)

  # Concatenate path gradients together row-wise into single tensor.
  total_gradients = tf.concat(gradient_batches, axis=0)

  # Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients)

  # Scale integrated gradients with respect to input.
  integrated_gradients = (instance - baseline) * avg_gradients

  return integrated_gradients


@tf.function(reduce_retracing=True)
def one_batch(baseline, instance, alpha_batch, model, target_class_idx):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_instances(baseline=baseline,
                                                       instance=instance,
                                                       alphas=alpha_batch)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(instances=interpolated_path_input_batch, model=model, target_class_idx=target_class_idx)
    return gradient_batch
