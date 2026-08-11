/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.parquet.hadoop;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.bytes.ByteBufferAllocator;
import org.apache.parquet.bytes.HeapByteBufferAllocator;
import org.apache.parquet.bytes.TrackingByteBufferAllocator;
import org.apache.parquet.column.ParquetProperties;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.filter2.compat.FilterCompat;
import org.apache.parquet.filter2.recordlevel.PhoneBookWriter;
import org.apache.parquet.hadoop.api.ReadSupport;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.hadoop.example.GroupReadSupport;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.io.DelegatingSeekableInputStream;
import org.apache.parquet.io.InputFile;
import org.apache.parquet.io.ParquetFileRange;
import org.apache.parquet.io.SeekableInputStream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Checks that the buffers a vectored read allocates are returned to the allocator.
 * <p>
 * Vectored IO is only wired up to a filesystem on Hadoop 3.3.5 and later, so these tests drive
 * {@link SeekableInputStream#readVectored(List, ByteBufferAllocator)} directly to stay meaningful
 * on every supported Hadoop version.
 */
public class TestVectoredIoBufferRelease {

  private static final Path FILE = createTempFile();
  private static final List<PhoneBookWriter.User> DATA = TestParquetReader.makeUsers(1000);

  @BeforeAll
  public static void createFile() throws IOException {
    int pageSize = DATA.size() / 10;
    int rowGroupSize = pageSize * 6 * 5;
    PhoneBookWriter.write(
        ExampleParquetWriter.builder(FILE)
            .withWriteMode(ParquetFileWriter.Mode.OVERWRITE)
            .withWriterVersion(ParquetProperties.WriterVersion.PARQUET_1_0)
            .withPageSize(pageSize)
            .withRowGroupSize(rowGroupSize),
        DATA);
  }

  @AfterAll
  public static void deleteFile() throws IOException {
    FILE.getFileSystem(new Configuration()).delete(FILE, false);
  }

  /**
   * A filesystem may complete the range futures with slices of a single merged allocation rather
   * than with the buffers it allocated, so both shapes have to release cleanly.
   */
  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  public void testVectoredReadReleasesAllocatedBuffers(boolean mergeRanges) throws IOException {
    Configuration conf = new Configuration();
    conf.setBoolean(ParquetInputFormat.HADOOP_VECTORED_IO_ENABLED, true);

    TrackingByteBufferAllocator allocator = TrackingByteBufferAllocator.wrap(new HeapByteBufferAllocator());
    VectoredInputFile file = new VectoredInputFile(HadoopInputFile.fromPath(FILE, conf), mergeRanges);

    List<PhoneBookWriter.User> users = PhoneBookWriter.readUsers(
        new GroupReaderBuilder()
            .withFile(file)
            .withConf(conf)
            .withAllocator(allocator)
            .withFilter(FilterCompat.NOOP),
        true);

    assertThat(users).hasSameSizeAs(DATA);
    assertThat(file.vectoredReads).isPositive();
    assertThatCode(allocator::close).doesNotThrowAnyException();
  }

  private static final class GroupReaderBuilder extends ParquetReader.Builder<Group> {

    @Override
    protected ReadSupport<Group> getReadSupport() {
      return new GroupReadSupport();
    }
  }

  private static Path createTempFile() {
    try {
      return new Path(Files.createTempFile("test-vectored-io_", ".parquet")
          .toAbsolutePath()
          .toString());
    } catch (IOException e) {
      throw new AssertionError("Unable to create temporary file", e);
    }
  }

  /**
   * Serves a stream that supports vectored reads whatever the Hadoop version on the classpath.
   */
  private static final class VectoredInputFile implements InputFile {

    private final InputFile delegate;
    private final boolean mergeRanges;
    private int vectoredReads;

    VectoredInputFile(InputFile delegate, boolean mergeRanges) {
      this.delegate = delegate;
      this.mergeRanges = mergeRanges;
    }

    @Override
    public long getLength() throws IOException {
      return delegate.getLength();
    }

    @Override
    public SeekableInputStream newStream() throws IOException {
      return new VectoredStream(delegate.newStream());
    }

    private final class VectoredStream extends DelegatingSeekableInputStream {

      private final SeekableInputStream stream;

      VectoredStream(SeekableInputStream stream) {
        super(stream);
        this.stream = stream;
      }

      @Override
      public long getPos() throws IOException {
        return stream.getPos();
      }

      @Override
      public void seek(long newPos) throws IOException {
        stream.seek(newPos);
      }

      @Override
      public boolean readVectoredAvailable(ByteBufferAllocator allocator) {
        return true;
      }

      @Override
      public void readVectored(List<ParquetFileRange> ranges, ByteBufferAllocator allocator) throws IOException {
        vectoredReads++;
        if (mergeRanges) {
          readAsSingleMergedRange(ranges, allocator);
        } else {
          readRangeByRange(ranges, allocator);
        }
      }

      /** One buffer per range, as {@code RawLocalFileSystem} does. */
      private void readRangeByRange(List<ParquetFileRange> ranges, ByteBufferAllocator allocator)
          throws IOException {
        for (ParquetFileRange range : ranges) {
          ByteBuffer buffer = allocator.allocate(range.getLength());
          stream.seek(range.getOffset());
          stream.readFully(buffer);
          buffer.flip();
          range.setDataReadFuture(CompletableFuture.completedFuture(buffer));
        }
      }

      /**
       * One buffer spanning every range, handing each range a slice of it. Object store
       * connectors merge nearby ranges this way, so the futures never see the allocation.
       */
      private void readAsSingleMergedRange(List<ParquetFileRange> ranges, ByteBufferAllocator allocator)
          throws IOException {
        long start = Long.MAX_VALUE;
        long end = Long.MIN_VALUE;
        for (ParquetFileRange range : ranges) {
          start = Math.min(start, range.getOffset());
          end = Math.max(end, range.getOffset() + range.getLength());
        }

        ByteBuffer merged = allocator.allocate((int) (end - start));
        stream.seek(start);
        stream.readFully(merged);
        merged.flip();

        List<ByteBuffer> slices = new ArrayList<>(ranges.size());
        for (ParquetFileRange range : ranges) {
          ByteBuffer slice = merged.duplicate();
          slice.position((int) (range.getOffset() - start));
          slice.limit((int) (range.getOffset() - start) + range.getLength());
          slices.add(slice.slice());
        }
        for (int i = 0; i < ranges.size(); i++) {
          ranges.get(i).setDataReadFuture(CompletableFuture.completedFuture(slices.get(i)));
        }
      }
    }
  }
}
