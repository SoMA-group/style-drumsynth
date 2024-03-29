# Style-based Drum Synthesis with GAN Inversion Demo
TensorFlow implementation of a style-based generative adversarial network for conditional synthesis of drum sounds.

![Overview of proposed style-based drum synthesis system](ADS.png)

## Audio examples

### Music created using generated drum sounds

* [Hip-hop track](https://soundcloud.com/beatsbygan/hip-hop-beat)
* [Drum and bass track](https://soundcloud.com/beatsbygan/drum-and-bass)
* [Breakbeat morphing track](https://soundcloud.com/beatsbygan/bb-morphing)

### Transforming input audio

* [Beatbox-to-drumgan](https://soundcloud.com/beatsbygan/beatbox-to-gan)
* [Breakbeat-to-drumgan](https://soundcloud.com/beatsbygan/hiphop-to-gan)


## Code

### Dependencies

#### Python

Code has been developed with `Python 3.6.13`. It should work with other versions of `Python 3`, but has not been tested. Moreover, we rely on several third-party libraries, listed in [`requirements.txt`](requirements.txt). They can be installed with

```bash
$ pip install -r requirements.txt
```

#### Checkpoints

The tensorflow checkpoints for loading pre-trained network weights can be download [here](https://drive.google.com/drive/folders/11v5-xXhPa6Rv6t5V2koeOM9MLdrxvWM9?usp=sharing). Unzip the folder and save it into this projects directory: "style-drumsynth/checkpoints".

### Usage

#### Synthesis

The code for generation is contained within the `drumsynth_demo.py` script, which enables conditional synthesises of drum sounds using a pretrained generator.

The following control parameters are available:
* Condition: which type of drum to generate? (kick, snare or hat) 
* Direction: "synthesis controls", which principal direction to move in? [0:4]
* Direction slider: how far to move in a particular direction?
* Number of generations: how many drums to generate?
* Stocastic Variation: amount of inconsequential noise to inject into the generator layers
* Randomize: generate by randomly sampling the latent space, or generate from a fixed, pre-computed latent vectors for a kick, snare and hat
* Encode: regenerate drum sounds stored in the style-drumsynth/input_audio
* Interpolation: demonstrates waveform interpolation between two drum sounds

Generations are saved in the style-drumsynth/generations folder.
To experiment encoding your own drums sounds, save your audio files in the "style-drumsynth/input_audio" directory.


#### drumsynth_demo.py arguments

```
  -c CONDITION,           --condition CONDITION
                            0: kick, 1: snare, 2:hat
  -d DIRECTION,           --direction DIRECTION
                            synthesis controls [0:4]
  -ds DIRECTION_SLIDER,   --direction_slider DIRECTION_SLIDER
                            how much to move in a particular direction
  -n NUM_GENERATIONS,     --num_generations NUM_GENERATIONS
                            number of examples to generate
  -v STOCASTIC_VARIATION, --stocastic_variation STOCASTIC_VARIATION
                            amount of inconsequential noise injected
  -r RANDOMIZE,           --randomize RANDOMIZE
                            if set to False, a fixed latent vector is used to generate a drum sound from each condition
  -e ENCODE,              --encode ENCODE
                            regenerates drum sounds from encoder folder (True/False)
  -i INTERPOLATION        --interpolation INTERPOLATION
                            creates linear path between two random latent vectors, generates drum sounds whilst moving along the path in steps                 
```

#### Training

The code for training is contained within the `train.py` script, which enables GAN training on a user defined dataset.

#### train.py main arguments

```
  -d DATASET,             --dataset DATASET
                            path to training data
  -p PREPROC,             --preproc PREPROC
                            set to True to prepare data for training. Ensure that drum sounds are split into class sub-folders (e.g. kicks, snares, cymbals)
  -cl N_CLASSES,          --n_classes N_CLASSES
  -sr SAMPLE_RATE,        --sr SAMPLE_RATE
  -z Z_DIM                --z_dim Z_DIM
  -b BATCH_SIZE           --batch_size BATCH_SIZE
  -e EPOCHS               --epochs EPOCHS
```



## Supporting webpage



For more information, please **visit the corresponding [supporting website](https://jake-drysdale.github.io/blog/stylegan-drumsynth/)**.

It contains the following:
  * Audio examples
  * Training data
  * Generations
  * Example usage within loop-based electronic music compositions
  * Generating Drum Loops
  * Interpolation demonstration
  * Supplementary figures
  * A link to the ISMIR LBD paper and post




## References

## References

| **[1]** |                  **[Drysdale, J. and Tomczak, M. and J. Hockman. 2021. Style-based Drum Synthesis with GAN Inversion. In *Extended Abstracts for the Late-Breaking Demo Sessions of the 22nd International Society for Music Information Retrieval Conference*, Online.](https://archives.ismir.net/ismir2021/latebreaking/000041.pdf)**|
| :---- | :--- |

```
@inproceedings{drysdale2021sds,
  title={Style-based Drum Synthesis with GAN Inversion},
  author={Drysdale, Jake and Tomczak, Maciej and Hockman, Jason},
  booktitle = {Extended Abstracts for the Late-Breaking Demo Sessions of the 22nd
  International Society for Music Information Retrieval (ISMIR) Conference.},
  year={2021}
}
```

## Help

Any questions please feel free to contact me on jake.drysdale@bcu.ac.uk


