from .utils import Timer, OffsetMapping, extract_entity


def decode(span_logits, pos_list: [], batch_data: [], tokenizer) -> []:
    # span_logits = torch.sigmoid(span_logits).cpu()
    # t0.print_time(f'after sigmoid and cpu(), .. batch_data size: {len(batch_data)}')
    # threshold = args.threshold
    # entity_idx_span_activations = torch.argwhere(span_logits > threshold)
    entity_idx_span_activations = pos_list
    # t0.print_time(f'after argwhere, .. batch_data size: {len(batch_data)}')

    for i, item in enumerate(batch_data):
        textb = item['text']
        offset_mapping = OffsetMapping().rematch(textb, tokenizer.tokenize(textb))

        # input_ids = tokenizer.encode('[SEP]' + textb, max_length=args.max_length, truncation='longest_first')

        for c in range(len(item['choices'])):

            texta = item['task_type'] + '[SEP]' + item['subtask_type'] + '[SEP]' + item['choices'][c]['entity_type']

            text_start_id = len(tokenizer.encode(texta))

            # logits = span_logits[i, c, :, :]
            # logits_cuda = span_logits_cuda[i, c, :, :]

            entity_name_list = []
            entity_list = []

            # sample_length = text_start_id + len(input_ids)
            # entity_idx_type_list = self.extract_index(logits, sample_length, split_value=args.threshold)
            entity_idx_type_list = []
            for entity_span in entity_idx_span_activations:
                if entity_span[0] == i and entity_span[1] == c:
                    prob = span_logits[i, c, entity_span[2], entity_span[3]].item()
                    entity_idx_type_list.append((entity_span[2], entity_span[3], prob))

            # entity_idx_type_list = self.extract_index(logits_cuda, sample_length, split_value=args.threshold)

            for entity_idx in entity_idx_type_list:

                entity = extract_entity(item['text'], (entity_idx[0], entity_idx[1]), text_start_id, offset_mapping)

                if entity not in entity_name_list:
                    entity_name_list.append(entity)

                    entity = {'entity_name': entity, 'score': entity_idx[2]}
                    # entity = {'entity_name': entity}
                    entity_list.append(entity)


            batch_data[i]['choices'][c]['entity_list'] = entity_list
    return batch_data
