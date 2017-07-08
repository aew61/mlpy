# SYSTEM IMPORTS



# PYTHON PROJECT IMPORTS


def create_training_and_validation_sets(examples, annotations, go_in_validation_set_func_ptr):
    assert(len(examples) == len(annotations))

    validation_examples = list()
    validation_annotations = list()

    training_examples = list()
    training_annotations = list()

    for example, annotation in zip(examples, annotations):
        if go_in_validation_set_func_ptr():
            validation_examples.append(example)
            validation_annotations.append(annotation)
        else:
            training_examples.append(example)
            training_annotations.append(example)

    assert(len(training_examples) == len(training_annotations))
    assert(len(validation_examples) == len(validation_annotations))
    assert(len(training_examples) + len(validation_examples) == len(examples))

    return (training_examples, training_annotations),\
           (validation_examples, validation_annotations)

def abstract_partition(examples, annotations, partition_scheme_func_ptr):
    assert(len(examples) == len(annotations))

    for example, annotation in zip(examples, annotations):
        partition_scheme_func_ptr(example, annotation)

