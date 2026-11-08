module attributes {flatppl.runtime_status = 2 : i64} {
  func.func @sample(%key: tensor<2xui64>, %arg0: tensor<3xf32>) -> (tensor<3xf32>, tensor<2xui64>, tensor<i32>) {
    %0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.minimum across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.compare LE, %1, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<9> : tensor<i32>
    %6 = stablehlo.select %3, %5, %4 : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<3xf32>
    %8 = stablehlo.constant dense<1.0> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %10 = stablehlo.compare LT, %7, %9 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
    %11 = stablehlo.add %7, %9 : tensor<3xf32>
    %12 = stablehlo.select %10, %11, %7 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %13 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %15 = stablehlo.subtract %12, %14 : tensor<3xf32>
    %16 = stablehlo.constant dense<9.0> : tensor<f32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %18 = stablehlo.multiply %17, %15 : tensor<3xf32>
    %19 = stablehlo.sqrt %18 : tensor<3xf32>
    %20 = stablehlo.divide %9, %19 : tensor<3xf32>
    %21, %22 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x3xui32>)
    %23 = stablehlo.constant dense<9> : tensor<128x3xui32>
    %24 = stablehlo.shift_right_logical %22, %23 : tensor<128x3xui32>
    %25 = stablehlo.convert %24 : (tensor<128x3xui32>) -> tensor<128x3xf32>
    %26 = stablehlo.constant dense<1.1920929E-7> : tensor<128x3xf32>
    %27 = stablehlo.multiply %25, %26 : tensor<128x3xf32>
    %28 = stablehlo.constant dense<2.0> : tensor<128x3xf32>
    %29 = stablehlo.constant dense<1.0> : tensor<128x3xf32>
    %30 = stablehlo.multiply %27, %28 : tensor<128x3xf32>
    %31 = stablehlo.subtract %30, %29 : tensor<128x3xf32>
    %32 = chlo.erf_inv %31 : tensor<128x3xf32> -> tensor<128x3xf32>
    %33 = stablehlo.constant dense<1.4142135623730951> : tensor<128x3xf32>
    %34 = stablehlo.multiply %32, %33 : tensor<128x3xf32>
    %35, %36 = stablehlo.rng_bit_generator %21, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x3xui32>)
    %37 = stablehlo.constant dense<9> : tensor<128x3xui32>
    %38 = stablehlo.shift_right_logical %36, %37 : tensor<128x3xui32>
    %39 = stablehlo.convert %38 : (tensor<128x3xui32>) -> tensor<128x3xf32>
    %40 = stablehlo.constant dense<1.1920929E-7> : tensor<128x3xf32>
    %41 = stablehlo.multiply %39, %40 : tensor<128x3xf32>
    %42 = stablehlo.constant dense<false> : tensor<3xi1>
    %43 = stablehlo.constant dense<0.0> : tensor<3xf32>
    %47:3 = stablehlo.while(%44 = %4, %45 = %42, %46 = %43) : tensor<i32>, tensor<3xi1>, tensor<3xf32>
    cond {
      %48 = stablehlo.constant dense<128> : tensor<i32>
      %49 = stablehlo.compare LT, %44, %48, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %50 = stablehlo.constant dense<true> : tensor<i1>
      %51 = stablehlo.reduce(%45 init: %50) applies stablehlo.and across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
      %52 = stablehlo.not %51 : tensor<i1>
      %53 = stablehlo.and %49, %52 : tensor<i1>
      stablehlo.return %53 : tensor<i1>
    } do {
      %54 = stablehlo.constant dense<0> : tensor<i32>
      %55 = stablehlo.dynamic_slice %34, %44, %54, sizes = [1, 3] : (tensor<128x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x3xf32>
      %56 = stablehlo.reshape %55 : (tensor<1x3xf32>) -> tensor<3xf32>
      %57 = stablehlo.dynamic_slice %41, %44, %54, sizes = [1, 3] : (tensor<128x3xf32>, tensor<i32>, tensor<i32>) -> tensor<1x3xf32>
      %58 = stablehlo.reshape %57 : (tensor<1x3xf32>) -> tensor<3xf32>
      %59 = stablehlo.multiply %20, %56 : tensor<3xf32>
      %60 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %61 = stablehlo.add %60, %59 : tensor<3xf32>
      %62 = stablehlo.multiply %61, %61 : tensor<3xf32>
      %63 = stablehlo.multiply %62, %61 : tensor<3xf32>
      %64 = stablehlo.multiply %15, %63 : tensor<3xf32>
      %65 = stablehlo.constant dense<0.5> : tensor<f32>
      %66 = stablehlo.multiply %56, %56 : tensor<3xf32>
      %67 = stablehlo.broadcast_in_dim %65, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %68 = stablehlo.multiply %67, %66 : tensor<3xf32>
      %69 = stablehlo.negate %64 : tensor<3xf32>
      %70 = stablehlo.log %63 : tensor<3xf32>
      %71 = stablehlo.multiply %15, %70 : tensor<3xf32>
      %72 = stablehlo.add %68, %15 : tensor<3xf32>
      %73 = stablehlo.add %72, %69 : tensor<3xf32>
      %74 = stablehlo.add %73, %71 : tensor<3xf32>
      %75 = stablehlo.log %58 : tensor<3xf32>
      %76 = stablehlo.compare LT, %75, %74 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %77 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<3xf32>
      %78 = stablehlo.compare GT, %63, %77 : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
      %79 = stablehlo.and %76, %78 : tensor<3xi1>
      %80 = stablehlo.select %45, %46, %64 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
      %81 = stablehlo.or %45, %79 : tensor<3xi1>
      %82 = stablehlo.constant dense<1> : tensor<i32>
      %83 = stablehlo.add %44, %82 : tensor<i32>
      stablehlo.return %83, %81, %80 : tensor<i32>, tensor<3xi1>, tensor<3xf32>
    }
    %84, %85 = stablehlo.rng_bit_generator %35, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<3xui32>)
    %86 = stablehlo.constant dense<9> : tensor<3xui32>
    %87 = stablehlo.shift_right_logical %85, %86 : tensor<3xui32>
    %88 = stablehlo.convert %87 : (tensor<3xui32>) -> tensor<3xf32>
    %89 = stablehlo.constant dense<1.1920929E-7> : tensor<3xf32>
    %90 = stablehlo.multiply %88, %89 : tensor<3xf32>
    %91 = stablehlo.log %47#2 : tensor<3xf32>
    %92 = stablehlo.log %90 : tensor<3xf32>
    %93 = stablehlo.divide %92, %7 : tensor<3xf32>
    %94 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %95 = stablehlo.select %10, %93, %94 : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %96 = stablehlo.add %91, %95 : tensor<3xf32>
    %97 = stablehlo.constant dense<"0x00000000"> : tensor<f32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %99 = stablehlo.subtract %96, %98 : tensor<3xf32>
    %100 = stablehlo.reshape %99 : (tensor<3xf32>) -> tensor<3xf32>
    %101 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %102 = stablehlo.reduce(%100 init: %101) applies stablehlo.maximum across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %103 = stablehlo.broadcast_in_dim %102, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %104 = stablehlo.subtract %100, %103 : tensor<3xf32>
    %105 = stablehlo.exponential %104 : tensor<3xf32>
    %106 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %107 = stablehlo.reduce(%105 init: %106) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
    %108 = stablehlo.broadcast_in_dim %107, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %109 = stablehlo.divide %105, %108 : tensor<3xf32>
    return %109, %84, %6 : tensor<3xf32>, tensor<2xui64>, tensor<i32>
  }
}
