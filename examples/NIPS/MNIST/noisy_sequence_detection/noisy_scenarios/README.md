This folder contains the code to generate the noisy scenarios.

There are the following types of noisy scenarios:

- **scenario103**: For each complex event in the training data, there is 
a probability that the label for the complex event will be substituted by
a random complex event label.

- **scenario104**: For each complex event type, a substitute is selected. When
that complex event happens in the training data, there is a probability that
the label will be changed for the pre-selected substitute.

- **scenario108**: For a pre-selected targeted complex event type, a substitute 
is selected. When the targeted complex event happens in the training data, there
is a probability that the label will be changed for the pre-selected substitute.

- **scenario109**: This attack targets a pre-selected complex event type (0 
by default). When the targeted complex event happens in the training data, 
there is a probability that the label will be changed for a randomly chosen 
complex event.

- **scenario110**: Same as scenario109 but the randomly chosen complex event 
used for the substitution cannot be the target. 