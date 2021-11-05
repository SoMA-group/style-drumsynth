# Style-based Drum Synthesis with GAN Inversion Demo
TensorFlow implementation of a style-based generative adversarial network for conditional synthesis of drum sounds.

![Overview of proposed style-based drum synthesis system](ADS.png)

## Audio examples

### Music created using generated drum sounds

* [Hip-hop track](https://soundcloud.com/beatsbygan/hip-hop-beat)
<iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/896706142&color=%23ff5500&auto_play=false&hide_related=true&show_comments=false&show_user=true&show_reposts=false&show_teaser=false&visual=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/beatsbygan" title="beatsbygan" target="_blank" style="color: #cccccc; text-decoration: none;">beatsbygan</a> · <a href="https://soundcloud.com/beatsbygan/hip-hop-beat" title="Hip hop beat" target="_blank" style="color: #cccccc; text-decoration: none;">Hip hop beat</a></div>
* [Drum and bass track](https://soundcloud.com/beatsbygan/drum-and-bass)
* [Breakbeat morphing track](https://soundcloud.com/beatsbygan/bb-morphing)

### Transforming input audio

* [Beatbox-to-drumgan](https://soundcloud.com/beatsbygan/beatbox-to-gan)
* [Breakbeat-to-drumgan](https://soundcloud.com/beatsbygan/hiphop-to-gan)


### This drum sound does not exist

Check out some randomly generated drum sound waveforms [here](https://tdsdne.vercel.app/)

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

The code for generation is contained within the `drumsynth_demo.py` script, which enables conditional synthesises of drum sounds using a pretrained generator.

The following control parameters are available:
* Condition: which type of drum to generate (kick, snare or hat) 
* Direction: "Synthesis controls", which principal direction to move in [0:4]
* Direction slider: How far to move in a particular direction
* Number of generations: How many drums to generate
* Stocastic Variation: Amount of inconsequential noise to inject into the generator
* Randomize: Generate by randomly sampling the latent space, or generate from a fixed, pre-computed latent vectors for a kick, snare and hat
* Encode: regenerate drum sounds stored in the style-drumsynth/input_audio


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
                            regenerates drum sounds from encoder folder
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
  * A link to the DAFx 2020 paper and presentation




## References

| **[1]** |                  **[Drysdale, J., M. Tomczak, J. Hockman, Adversarial Synthesis of Drum Sounds. Proceedings of the 23rd International Conference on Digital Audio Effects (DAFX), 2020.](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx2020_paper_45.pdf)**|
| :---- | :--- |

```
@inproceedings{drysdale2020ads,
  title={Adversarial synthesis of drum sounds},
  author={Drysdale, Jake and Tomczak, Maciek and Hockman, Jason},
  booktitle = {Proceedings of the International Conference on Digital Audio Effects (DAFx)},
  year={2020}
}
```


## Help

Any questions please feel free to contact me on jake.drysdale@bcu.ac.uk


