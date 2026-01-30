import React from 'react';
import { GlassCard } from './ui/GlassCard';
import { GlassInput } from './ui/GlassInput';
import { GlassSelect } from './ui/GlassSelect';
import { useTranslation } from 'react-i18next';
import { AlertTriangle } from 'lucide-react';

export interface TrainingConfigProps {
    data: any;
    modelType?: string;
    onChange: (data: any) => void;
    validationEnabled?: boolean;
}

export function TrainingConfig({ data, modelType, onChange, validationEnabled = true }: TrainingConfigProps) {
    const { t } = useTranslation();

    const isVideoModel = ['hunyuan_video', 'ltx_video', 'wan21', 'wan22', 'hunyuan_video_15', 'cosmos'].includes(modelType || '');

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        onChange({ ...data, [e.target.name]: e.target.value });
    };

    return (
        <div className="space-y-6">
            <GlassCard className="p-6">
                <div className="mb-6">
                    <h3 className="text-2xl font-bold">{t('training.title')}</h3>
                    <p className="text-sm text-muted-foreground">{t('training.desc')}</p>
                </div>

                <div className="grid gap-6 md:grid-cols-3">
                    <GlassInput label={t('training.output_name')} name="output_folder_name" value={data.output_folder_name ?? 'mylora'} onChange={handleChange} />
                    <GlassInput label={t('training.epochs')} name="epochs" type="number" value={data.epochs ?? 50} onChange={handleChange} />
                    <GlassInput label={t('training.batch_size')} name="micro_batch_size_per_gpu" type="number" value={data.micro_batch_size_per_gpu ?? 1} onChange={handleChange} />
                    <GlassInput label={t('training.image_micro_batch_size_per_gpu')} name="image_micro_batch_size_per_gpu" type="number" value={data.image_micro_batch_size_per_gpu ?? data.micro_batch_size_per_gpu ?? 1} onChange={handleChange} placeholder={t('common.optional')} />

                    <GlassInput label={t('training.grad_accumulation')} name="gradient_accumulation_steps" type="number" value={data.gradient_accumulation_steps ?? 3} onChange={handleChange} />
                    <GlassInput label={t('training.warmup_steps')} name="warmup_steps" type="number" min="0" value={data.warmup_steps ?? 500} onChange={handleChange} />

                    <GlassSelect
                        label={t('training.lr_scheduler')}
                        name="lr_scheduler"
                        value={data.lr_scheduler ?? 'linear'}
                        onChange={handleChange}
                        options={[
                            { label: 'Constant', value: 'constant' },
                            { label: 'Linear', value: 'linear' }
                        ]}
                    />

                    <GlassInput label={t('training.grad_clipping')} name="gradient_clipping" type="number" step="0.1" value={data.gradient_clipping ?? 1.0} onChange={handleChange} />

                    <GlassSelect
                        label={t('training.save_dtype')}
                        name="save_dtype"
                        value={data.save_dtype ?? 'bfloat16'}
                        onChange={handleChange}
                        options={[{ label: 'bfloat16', value: 'bfloat16' }, { label: 'float16', value: 'float16' }, { label: 'float32', value: 'float32' }]}
                    />

                    <GlassSelect
                        label={t('training.partition_method')}
                        name="partition_method"
                        value={data.partition_method ?? 'parameters'}
                        onChange={handleChange}
                        options={[{ label: 'parameters', value: 'parameters' }, { label: 'uniform', value: 'uniform' }, { label: 'memory', value: 'memory' }]}
                    />

                    <GlassSelect
                        label={t('training.activation_checkpointing')}
                        name="activation_checkpointing"
                        value={data.activation_checkpointing ?? 'true'}
                        onChange={handleChange}
                        options={[
                            { label: t('dataset.enabled'), value: 'true' },
                            { label: t('dataset.disabled'), value: 'false' },
                            { label: t('training.activation_checkpointing_unsloth'), value: 'unsloth' }
                        ]}
                    />

                    <GlassInput label={t('training.pipeline_stages')} name="pipeline_stages" type="number" value={data.pipeline_stages ?? 1} onChange={handleChange} />
                    <GlassInput label={t('training.blocks_to_swap')} name="blocks_to_swap" type="number" min={0} value={data.blocks_to_swap ?? 0} onChange={handleChange} />
                    <GlassInput label={t('training.caching_batch_size')} name="caching_batch_size" type="number" value={data.caching_batch_size ?? 1} onChange={handleChange} />

                    {isVideoModel && (
                        <GlassSelect
                            label={t('training.video_clip_mode')}
                            name="video_clip_mode"
                            value={data.video_clip_mode ?? 'none'}
                            onChange={handleChange}
                            options={[
                                { label: 'None', value: 'none' },
                                { label: 'Single Beginning', value: 'single_beginning' },
                                { label: 'Single Middle', value: 'single_middle' },
                                { label: 'Multiple Overlapping', value: 'multiple_overlapping' }
                            ]}
                        />
                    )}

                    <GlassInput label={t('training.steps_per_print')} name="steps_per_print" type="number" value={data.steps_per_print ?? 1} onChange={handleChange} />
                    <GlassInput label={t('training.save_every_n_epochs')} name="save_every_n_epochs" type="number" value={data.save_every_n_epochs ?? 1} onChange={handleChange} />
                    <GlassInput label={t('training.checkpoint_every_n_minutes')} name="checkpoint_every_n_minutes" type="number" value={data.checkpoint_every_n_minutes ?? 120} onChange={handleChange} />

                    <div className="md:col-span-3 border-t border-white/10 my-4 pt-4">
                        <h4 className="text-lg font-semibold mb-4 text-muted-foreground">{t('training.eval_settings')}</h4>

                        {!validationEnabled ? (
                            <div className="p-4 rounded-xl bg-white/5 border border-white/10 flex items-center gap-3 text-yellow-500/80">
                                <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                                <span className="text-sm">{t('training.validation_disabled_hint')}</span>
                            </div>
                        ) : (
                            <div className="grid gap-6 md:grid-cols-3">
                                <GlassInput label={t('training.eval_every_n_epochs')} name="eval_every_n_epochs" type="number" value={data.eval_every_n_epochs ?? 1} onChange={handleChange} />
                                <GlassInput label={t('training.eval_batch_size')} name="eval_micro_batch_size_per_gpu" type="number" value={data.eval_micro_batch_size_per_gpu ?? 1} onChange={handleChange} />
                                <GlassInput label={t('training.image_eval_micro_batch_size_per_gpu')} name="image_eval_micro_batch_size_per_gpu" type="number" value={data.image_eval_micro_batch_size_per_gpu ?? data.eval_micro_batch_size_per_gpu ?? 1} onChange={handleChange} placeholder={t('common.optional')} />
                                <GlassInput label={t('training.eval_grad_accumulation')} name="eval_gradient_accumulation_steps" type="number" value={data.eval_gradient_accumulation_steps ?? 1} onChange={handleChange} />

                                <div className="flex items-center gap-2 mt-8">
                                    <input type="checkbox" name="eval_before_first_step" className="w-4 h-4" checked={data.eval_before_first_step !== false} onChange={(e) => onChange({ ...data, eval_before_first_step: e.target.checked })} />
                                    <label className="text-sm">{t('training.eval_before_first_step')}</label>
                                </div>
                                <div className="flex items-center gap-2 mt-8">
                                    <input type="checkbox" name="disable_block_swap_for_eval" className="w-4 h-4" checked={!!data.disable_block_swap_for_eval} onChange={(e) => onChange({ ...data, disable_block_swap_for_eval: e.target.checked })} />
                                    <label className="text-sm">{t('training.disable_block_swap_for_eval')}</label>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </GlassCard>
        </div>
    );
}
