import React from 'react';
import { GlassCard } from './ui/GlassCard';
import { GlassInput } from './ui/GlassInput';
import { useTranslation } from 'react-i18next';

export interface AdvancedTrainingConfigProps {
    data: any;
    onChange: (data: any) => void;
}

export function AdvancedTrainingConfig({ data, onChange }: AdvancedTrainingConfigProps) {
    const { t } = useTranslation();

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const value = e.target.type === 'checkbox' ? (e.target as HTMLInputElement).checked : e.target.value;
        onChange({ ...data, [e.target.name]: value });
    };

    return (
        <GlassCard className="p-6">
            <div className="mb-6">
                <h3 className="text-2xl font-bold">{t('advanced_training.title')}</h3>
                <p className="text-sm text-muted-foreground">{t('advanced_training.desc')}</p>
            </div>

            <div className="grid gap-6 md:grid-cols-3">
                <GlassInput label={t('advanced_training.max_steps')} name="max_steps" type="number" value={data.max_steps ?? 0} onChange={handleChange} />
                <GlassInput label={t('advanced_training.force_constant_lr')} name="force_constant_lr" type="number" step="1e-8" value={data.force_constant_lr ?? 0.0} onChange={handleChange} />

                <GlassInput label={t('advanced_training.pseudo_huber_c')} name="pseudo_huber_c" type="number" step="0.1" value={data.pseudo_huber_c ?? 0.0} onChange={handleChange} />
                <GlassInput label={t('advanced_training.map_num_proc')} name="map_num_proc" type="number" value={data.map_num_proc ?? 0} onChange={handleChange} />

                <GlassInput label={t('advanced_training.save_every_n_steps')} name="save_every_n_steps" type="number" value={data.save_every_n_steps ?? 0} onChange={handleChange} />
                <GlassInput label={t('advanced_training.checkpoint_every_n_epochs')} name="checkpoint_every_n_epochs" type="number" value={data.checkpoint_every_n_epochs ?? 0} onChange={handleChange} />

                <div className="md:col-span-3">
                    <GlassInput label={t('advanced_training.partition_split')} name="partition_split" value={data.partition_split ?? ''} onChange={handleChange} placeholder="10,20" />
                </div>

                <div className="flex items-center gap-2 mt-2">
                    <input type="checkbox" name="compile" className="w-4 h-4" checked={!!data.compile} onChange={handleChange} />
                    <label className="text-sm">{t('advanced_training.compile')}</label>
                </div>

                <div className="flex items-center gap-2 mt-2">
                    <input type="checkbox" name="x_axis_examples" className="w-4 h-4" checked={!!data.x_axis_examples} onChange={handleChange} />
                    <label className="text-sm">{t('advanced_training.x_axis_examples')}</label>
                </div>

                <div className="flex items-center gap-2 mt-2">
                    <input type="checkbox" name="reentrant_activation_checkpointing" className="w-4 h-4" checked={!!data.reentrant_activation_checkpointing} onChange={handleChange} />
                    <label className="text-sm">{t('advanced_training.reentrant_activation_checkpointing')}</label>
                </div>
            </div>
        </GlassCard>
    );
}
