import theano
from theano import tensor as T

def rnn(step_function, 
        inputs,
        nb_other_outputs,
        initial_states,
        contexts,
        parameters,
        truncate_gradient=-1,
        go_backwards=False, 
        masking=True):
    '''Iterate over the time dimension of a tensor.

    Parameters
    ----------
    inputs: tensor of temporal data of shape (samples, time, ...)
        (at least 3D).
    step_function:
        Parameters:
            input: tensor with shape (samples, ...) (no time dimension),
                representing input for the batch of samples at a certain
                time step.
            states: list of tensors.
        Returns:
            output: tensor with shape (samples, ...) (no time dimension),
            new_states: list of tensors, same length and shapes
                as 'states'.
    initial_states: tensor with shape (samples, ...) (no time dimension),
        containing the initial values for the states used in
        the step function.
    contexts: list of contexts that are passed to step_function at each 
        steps.
    parameters: list of parameters that are passed to step_function at
        each steps.
    truncate_gradient
        ``truncate_gradient`` is the number of steps to use in truncated
        BPTT.  If you compute gradients through a scan op, they are
        computed using backpropagation through time. By providing a
        different value then -1, you choose to use truncated BPTT instead
        of classical BPTT, where you go for only ``truncate_gradient``
        number of steps back in time.
    go_backwards: boolean. If True, do the iteration over
        the time dimension in reverse order.
    masking: boolean. If true, any input timestep inputs[s, i]
        that is all-zeros will be skipped (states will be passed to
        the next step unchanged) and the corresponding output will
        be all zeros.

    Returns
    -------
    A tuple (last_output, outputs, new_states).
        last_output: the latest output of the rnn, of shape (samples, ...)
        outputs: tensor with shape (samples, time, ...) where each
            entry outputs[s, t] is the output of the step function
            at time t for sample s.
        new_states: list of tensors, latest states returned by
            the step function, of shape (samples, ...).
    '''
    inputs = inputs.dimshuffle((1, 0, 2))
    nb_states = len(initial_states)

    def _step(input, *args): 
        # separate states and contexts
        states = args[0:nb_states]
        output, other_outputs, new_states = step_function(input, args)
        if masking:
            # if all-zero input timestep, return
            # all-zero output and unchanged states
            switch = T.any(input, axis=-1, keepdims=True)
            output = T.switch(switch, output, 0. * output)
            for other_output in other_outputs:
                other_output = T.switch(switch, other_output, 0. * other_output)      
            return_states = []
            for state, new_state in zip(states, new_states):
                return_states.append(T.switch(switch, new_state, state))
            return [output] + other_outputs + return_states
        else:
            return [output] + other_outputs + new_states

    results, _ = theano.scan(
        _step,
        sequences=inputs,
        outputs_info=[None]*(1+nb_other_outputs) + initial_states,
        no_sequence=contexts + parameters,
        truncate_gradient=truncate_gradient,
        go_backwards=go_backwards)

    # deal with Theano API inconsistency
    if type(results) is list:
        outputs = results[0]
        other_outputs = results[1:1+nb_other_outputs]
        states = results[1+nb_other_outputs:]
    else:
        outputs = results
        other_outputs = []
        states = []

    outputs = T.squeeze(outputs)
    last_output = outputs[-1]

    outputs = outputs.dimshuffle((1, 0, 2))
    states = [T.squeeze(state[-1]) for state in states]
    return last_output, outputs, other_outputs, states


