import sys,json
for l in sys.stdin:
    try: r=json.loads(l)
    except: print(l,end=''); continue
    print(f"{r['name']:36s} pts={r['val_points']:6.2f} drift={r['R_drift']:.3f} shared={r['R_shared_change']:.3f} short={r['R_short_history']:.3f} stable={r['R_stable']:.3f} trainBrier={r['train_brier']:.4f}")
