This project can be compiled either using Maven directly or the Makefile provided.

Run class:   `ecs.Main`

Arguments:  `<run-id> <training-path> [<test-path> [quiet]]`

Evaluate
========

To train on 50% of the training set (actually 50% of each class, assuming equal classes)
and then evaluate, run:

    ecs.Main <run-id> <training-path>

It prints the result and debug information on stdout.

Generate submission file
========================

To train on all the data in the training set and generate the submission file
with predicted classes, run:

    ecs.Main <run-id> <training-path> <test-path> [quiet]

    The quiet parameter is optional. If given, it outputs the progress of predictions
    on standard error. Non-quiet version best used with output redirection in the command line.

Valid `<run-id>`
================

- run1
- run2
- run3
- random

random is a random classifier, was used for testing the Main class.

Paths
=====

The paths must be in URI format, as accepted by VFS.

:)
