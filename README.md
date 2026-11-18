# Mixamo FBX to ARMA 3 Animation Converter

It is what it says. With help from Claude, because although I do this stuff for a living I wanted to see if the tedious task of rig remapping could be handled by a decent LLM. Turns out, it can!

# Usage

### Pivots are Important

This is the source of the ARMA 3 rigs pivot names. Use the one they give you in the ARMA 3 Samples download.

The `--root extract` flag is useful for non-idle animations. It will make idle anumations do funny things sometimes.

```
:: One file -> Walking.rtm next to it
blender -b --python mix2arma.py -- --pivots "<Samples>\Addons\TemplateRTM\ManSkeleton_Pivots.p3d" -i Walking.fbx

:: One file -> specific output path
blender -b --python mix2arma.py -- --pivots "<Samples>\Addons\TemplateRTM\ManSkeleton_Pivots.p3d" -i Walking.fbx -o anims\walk_slow.rtm

:: Whole folder -> RTMs next to each FBX
blender -b --python mix2arma.py -- --pivots "<Samples>\Addons\TemplateRTM\ManSkeleton_Pivots.p3d" -i fbx

:: Whole folder -> separate output folder
blender -b --python mix2arma.py -- --pivots "<Samples>\Addons\TemplateRTM\ManSkeleton_Pivots.p3d" -i fbx -o rtm
```